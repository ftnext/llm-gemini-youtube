import json
import logging
import os
from unittest.mock import patch

import llm
from google.cloud import storage
from google.genai import Client, types


# Suppress an upstream AFC warning emitted even when no tools are configured.
# https://github.com/googleapis/python-genai/issues/2902
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


SUPPORTED_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
)


@llm.hookimpl
def register_models(register):
    for model_id in SUPPORTED_MODELS:
        register(GeminiYouTube(f"{model_id}-yt"))
        register(GeminiYouTubeBatch(f"{model_id}-yt-batch"))


def is_youtube_uri(url: str) -> bool:
    return (
        "youtube.com/watch?v=" in url
        or "youtu.be/" in url
        or "youtube.com/shorts/" in url
    )


def is_google_cloud_enabled() -> bool:
    enterprise = os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE")
    if enterprise is not None:
        return enterprise.lower() in {"true", "1"}

    vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
    return vertexai is not None and vertexai.lower() in {"true", "1"}


def require_google_cloud_for_batch() -> None:
    if not is_google_cloud_enabled():
        raise llm.ModelError(
            "Batch submission requires Google Cloud. Set "
            "GOOGLE_GENAI_USE_ENTERPRISE or GOOGLE_GENAI_USE_VERTEXAI "
            "to enable it."
        )


def parse_batch_input_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise llm.ModelError(
            "batch_input_uri must be a gs:// URI such as "
            "'gs://bucket/path/input.jsonl'"
        )
    bucket, _, object_name = uri.removeprefix("gs://").partition("/")
    if not bucket or not object_name:
        raise llm.ModelError(
            "batch_input_uri must name a bucket and an object, such as "
            "'gs://bucket/path/input.jsonl'"
        )
    if not object_name.endswith(".jsonl"):
        raise llm.ModelError("batch_input_uri must point to a .jsonl object")
    return bucket, object_name


def build_batch_request(video_uri: str, prompt_text: str) -> dict:
    return {
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "fileUri": video_uri,
                                "mimeType": "video/mp4",
                            },
                            "mediaProcessing": "AGENTIC",
                        },
                        {"text": prompt_text},
                    ],
                }
            ]
        }
    }


class GeminiYouTube(llm.KeyModel):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = True
    attachment_types = set(["text/html; charset=utf-8"])  # for YouTube URLs

    def __init__(self, model_id: str):
        self.model_id = model_id

    def get_key(self, explicit_key=None):
        if is_google_cloud_enabled():
            return None
        return super().get_key(explicit_key)

    def execute(self, prompt, stream, response, conversation, key):
        if not prompt.attachments:
            raise llm.ModelError("Attachment (YouTube URL) is required.")

        youtube_uri = None
        for attachment in prompt.attachments:
            if attachment.url and is_youtube_uri(attachment.url):
                youtube_uri = attachment.url
                break
        if not youtube_uri:
            raise llm.ModelError("YouTube URL attachment is required.")

        # Let google-genai select the Gemini Developer API or Gemini Enterprise
        # Agent Platform from its standard environment variables. The LLM key
        # is exposed only while Client reads its configuration.
        if key:
            with patch.dict(os.environ, {"GOOGLE_API_KEY": key}):
                client = Client()
        else:
            client = Client()

        model_id = self.model_id.removesuffix("-yt")
        if is_google_cloud_enabled():
            video_part = types.Part(
                file_data=types.FileData(
                    file_uri=youtube_uri,
                    mime_type="video/mp4",
                ),
                media_processing="AGENTIC",
            )
            contents = [video_part, prompt.prompt]

            if not stream:
                result = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                )
                yield result.text or ""
                return

            for chunk in client.models.generate_content_stream(
                model=model_id,
                contents=contents,
            ):
                if chunk.text:
                    yield chunk.text
            return

        interaction = client.interactions.create(
            model=model_id,
            input=[
                {
                    "type": "video",
                    "uri": youtube_uri,
                    "processing": "agentic",
                },
                {"type": "text", "text": prompt.prompt},
            ],
            stream=stream,
        )

        if not stream:
            yield interaction.output_text or ""
            return

        for event in interaction:
            if event.event_type == "step.delta" and event.delta.type == "text":
                yield event.delta.text


class GeminiYouTubeBatch(GeminiYouTube):
    supports_conversation = False

    class Options(llm.Options):
        batch_input_uri: str | None = None

    def get_key(self, explicit_key=None):
        require_google_cloud_for_batch()
        return None

    def execute(self, prompt, stream, response, conversation, key):
        require_google_cloud_for_batch()

        if prompt.system:
            raise llm.ModelError(
                "Batch models do not support system prompts."
            )

        messages = prompt.messages
        if len(messages) > 1 or (messages and messages[0].role != "user"):
            raise llm.ModelError(
                "Batch models do not support conversation history."
            )

        if not prompt.attachments:
            raise llm.ModelError("Attachment (YouTube URL) is required.")
        for attachment in prompt.attachments:
            if not attachment.url or not is_youtube_uri(attachment.url):
                raise llm.ModelError(
                    "All attachments must be YouTube URLs."
                )

        batch_input_uri = getattr(prompt.options, "batch_input_uri", None)
        if not batch_input_uri:
            raise llm.ModelError(
                "The batch_input_uri option is required: "
                "-o batch_input_uri 'gs://bucket/path/input.jsonl'"
            )
        bucket_name, object_name = parse_batch_input_uri(batch_input_uri)

        lines = [
            json.dumps(
                build_batch_request(attachment.url, prompt.prompt),
                ensure_ascii=False,
            )
            for attachment in prompt.attachments
        ]

        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_name).blob(object_name)
        blob.upload_from_string("\n".join(lines))

        client = Client(http_options=types.HttpOptions(api_version="v1beta1"))
        job = client.batches.create(
            model=self.model_id.removesuffix("-yt-batch"),
            src=batch_input_uri,
        )
        yield job.name

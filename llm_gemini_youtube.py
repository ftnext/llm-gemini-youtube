import json
import logging
import os
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4

import llm
from google.genai import Client, types
from google.genai.errors import APIError
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from google.cloud import storage
from pydantic import Field, Json


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
    register(GeminiVideoBatch("gemini-3.8-flash-batch"))


def video_part(uri):
    return types.Part(
        file_data=types.FileData(file_uri=uri, mime_type="video/mp4"),
        media_processing="AGENTIC",
    )


def parse_gcs_uri(uri):
    parsed = urlsplit(uri)
    if (parsed.scheme != "gs" or not parsed.netloc or parsed.query
            or parsed.fragment or "@" in parsed.netloc or ":" in parsed.netloc):
        raise llm.ModelError("Expected a Cloud Storage URI: gs://bucket/path")
    return parsed.netloc, parsed.path.lstrip("/")


def batch_jsonl(videos, text):
    if not videos:
        raise llm.ModelError("At least one video is required.")
    rows = []
    for uri in videos:
        parsed = urlsplit(uri)
        if not (
            parsed.scheme in {"https", "http"}
            and parsed.hostname in {
                "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"
            }
            and is_youtube_uri(uri)
        ):
            raise llm.ModelError("Batch videos must be YouTube URLs.")
        content = types.Content(
            role="user", parts=[video_part(uri), types.Part(text=text or "")]
        )
        rows.append(json.dumps(
            {"request": {"contents": [content.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )]}}, ensure_ascii=False,
        ))
    return "\n".join(rows) + "\n"


class GeminiVideoBatch(llm.Model):
    """Submit one job per prompt and return its receipt as JSON text."""

    can_stream = False
    attachment_types = {"text/html; charset=utf-8", "text/html"}

    class Options(llm.Options):
        videos: Json[list[str]] | list[str] | None = Field(
            default=None,
            description="JSON array of YouTube URLs; one request per video."
        )
        input_prefix: str | None = Field(
            default=None, description="GCS prefix for JSONL uploads; overrides LLM_GEMINI_BATCH_INPUT_PREFIX."
        )
        output_prefix: str | None = Field(
            default=None, description="GCS prefix for results; overrides LLM_GEMINI_BATCH_OUTPUT_PREFIX."
        )

    def __init__(self, model_id):
        self.model_id = model_id

    def execute(self, prompt, stream, response, conversation):
        if not is_google_cloud_enabled():
            raise llm.ModelError("Batch mode requires GOOGLE_GENAI_USE_ENTERPRISE=true and Google Cloud ADC.")
        if not os.environ.get("GOOGLE_CLOUD_PROJECT") or not os.environ.get("GOOGLE_CLOUD_LOCATION"):
            raise llm.ModelError("Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION for batch mode.")
        if prompt.system or (conversation and conversation.responses):
            raise llm.ModelError("Batch mode accepts independent prompts, without system prompts or conversation history.")
        options = prompt.options
        input_prefix = options.input_prefix or os.environ.get("LLM_GEMINI_BATCH_INPUT_PREFIX")
        output_prefix = options.output_prefix or os.environ.get("LLM_GEMINI_BATCH_OUTPUT_PREFIX")
        if not input_prefix or not output_prefix:
            raise llm.ModelError("Set input_prefix and output_prefix options or LLM_GEMINI_BATCH_INPUT_PREFIX and LLM_GEMINI_BATCH_OUTPUT_PREFIX.")
        parse_gcs_uri(input_prefix)
        parse_gcs_uri(output_prefix)
        videos = list(options.videos or [])
        for attachment in prompt.attachments:
            if not attachment.url:
                raise llm.ModelError("Batch attachments must be YouTube URLs.")
            videos.append(attachment.url)
        payload = batch_jsonl(videos, prompt.prompt)
        submission_id = uuid4().hex
        input_uri = f"{input_prefix.rstrip('/')}/{submission_id}.jsonl"
        output_uri = f"{output_prefix.rstrip('/')}/{submission_id}/"
        bucket, blob = parse_gcs_uri(input_uri)
        model = self.model_id.removesuffix("-batch")
        stage = "upload input"
        try:
            with storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"]) as storage_client:
                storage_client.bucket(bucket).blob(blob).upload_from_string(
                    payload, content_type="application/jsonl", if_generation_match=0,
                )
            stage = "create job (check Cloud Console before retrying)"
            with Client() as client:
                job = client.batches.create(
                    model=model, src=input_uri,
                    config=types.CreateBatchJobConfig(dest=output_uri),
                )
        except (APIError, GoogleAPIError, GoogleAuthError, ValueError, OSError) as ex:
            raise llm.ModelError(
                f"Batch {stage} failed; input={input_uri}, output={output_uri}: {ex}"
            ) from ex
        receipt = {
            "mode": "batch", "name": job.name,
            "state": job.state.value if job.state is not None else None,
            "model": model, "input_uri": input_uri,
            "output_uri": output_uri, "request_count": len(videos),
        }
        response.response_json = receipt
        yield json.dumps(receipt, ensure_ascii=False)


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
            contents = [video_part(youtube_uri), prompt.prompt]

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

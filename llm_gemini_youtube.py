import os
from unittest.mock import patch

import llm
from google.genai import Client, types


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

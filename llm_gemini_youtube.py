import logging
import os
from typing import Literal, Optional
from unittest.mock import patch

import llm
from google.genai import Client, types
from pydantic import Field


# Suppress an upstream AFC warning emitted even when no tools are configured.
# https://github.com/googleapis/python-genai/issues/2902
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


SUPPORTED_MODELS = (
    "gemini-4-carbon",
    "gemini-4-neon",
    "gemini-4-argon",
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
)

RESOLUTION_LEVELS = {
    "low": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
    "medium": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM,
    "high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
}


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


def _get_resolution(prompt) -> Optional[str]:
    """Extract the optional resolution setting from prompt options."""
    options = getattr(prompt, "options", None)
    if options is None:
        return None
    if isinstance(options, dict):
        return options.get("resolution")
    return getattr(options, "resolution", None)


class GeminiYouTube(llm.KeyModel):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = True
    attachment_types = set(["text/html; charset=utf-8"])  # for YouTube URLs

    class Options(llm.Options):
        resolution: Optional[Literal["low", "medium", "high"]] = Field(
            default=None,
            description="Video media resolution for Google Cloud requests (low, medium, high).",
        )

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

        resolution = _get_resolution(prompt)
        google_cloud_enabled = is_google_cloud_enabled()
        if resolution is not None and resolution not in RESOLUTION_LEVELS:
            raise llm.ModelError(
                f"Unsupported resolution: {resolution}. Choose from low, medium, or high."
            )
        if resolution is not None and not google_cloud_enabled:
            raise llm.ModelError(
                "The 'resolution' option is currently only supported when Google Cloud is enabled."
            )

        # Let google-genai select the Gemini Developer API or Gemini Enterprise
        # Agent Platform from its standard environment variables. The LLM key
        # is exposed only while Client reads its configuration.
        if key:
            with patch.dict(os.environ, {"GOOGLE_API_KEY": key}):
                client = Client()
        else:
            client = Client()

        model_id = self.model_id.removesuffix("-yt")
        if google_cloud_enabled:
            part_kwargs = {
                "file_data": types.FileData(
                    file_uri=youtube_uri,
                    mime_type="video/mp4",
                ),
                "media_processing": "AGENTIC",
            }
            if resolution is not None:
                part_kwargs["media_resolution"] = types.PartMediaResolution(
                    level=RESOLUTION_LEVELS[resolution]
                )
            video_part = types.Part(**part_kwargs)
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

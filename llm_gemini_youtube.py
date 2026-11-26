import logging
import os
from unittest.mock import patch

import llm
from google.genai import Client, types


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


def _combine_token_counts(base: int | None, extra: int | None) -> int | None:
    """Combine a base token count with an optional extra count."""
    if base is None:
        return extra if extra else None
    return base + (extra or 0)


def _usage_to_dict(usage) -> dict:
    """Convert an SDK usage model or stub into a plain JSON-compatible dict."""
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    if isinstance(usage, dict):
        return {key: value for key, value in usage.items() if value is not None}
    return {
        key: value
        for key, value in vars(usage).items()
        if not key.startswith("_") and value is not None
    }


def _record_usage(
    response,
    usage,
    *,
    input_key: str,
    tool_use_key: str,
    output_key: str,
    thought_key: str,
    cached_key: str,
    total_key: str,
) -> None:
    """Record normalized input/output counts and extra details on llm.Response."""
    if response is None or usage is None:
        return

    details = _usage_to_dict(usage)
    prompt_tokens = details.get(input_key)
    tool_use_tokens = details.get(tool_use_key)
    output_tokens_raw = details.get(output_key)
    thought_tokens = details.get(thought_key)

    # Cached tokens are already included in prompt/input totals by Gemini; only
    # add tool-use prompt tokens to input and thinking tokens to output.
    input_tokens = _combine_token_counts(prompt_tokens, tool_use_tokens)
    output_tokens = _combine_token_counts(output_tokens_raw, thought_tokens)

    details.pop(total_key, None)
    if not tool_use_tokens:
        details.pop(input_key, None)
        details.pop(tool_use_key, None)
    if not thought_tokens:
        details.pop(output_key, None)
        details.pop(thought_key, None)
    if not details.get(cached_key):
        details.pop(cached_key, None)

    if input_tokens is None and output_tokens is None and not details:
        return

    response.set_usage(
        input=input_tokens,
        output=output_tokens,
        details=details or None,
    )


def _set_generate_content_usage(response, usage_metadata) -> None:
    """Record usage from GenerateContentResponse.usage_metadata."""
    _record_usage(
        response,
        usage_metadata,
        input_key="prompt_token_count",
        tool_use_key="tool_use_prompt_token_count",
        output_key="candidates_token_count",
        thought_key="thoughts_token_count",
        cached_key="cached_content_token_count",
        total_key="total_token_count",
    )


def _set_interaction_usage(response, usage) -> None:
    """Record usage from Interactions API Interaction.usage."""
    _record_usage(
        response,
        usage,
        input_key="total_input_tokens",
        tool_use_key="total_tool_use_tokens",
        output_key="total_output_tokens",
        thought_key="total_thought_tokens",
        cached_key="total_cached_tokens",
        total_key="total_tokens",
    )


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
                _set_generate_content_usage(
                    response, getattr(result, "usage_metadata", None)
                )
                yield result.text or ""
                return

            usage_metadata = None
            for chunk in client.models.generate_content_stream(
                model=model_id,
                contents=contents,
            ):
                chunk_usage = getattr(chunk, "usage_metadata", None)
                if chunk_usage is not None:
                    usage_metadata = chunk_usage
                if chunk.text:
                    yield chunk.text
            _set_generate_content_usage(response, usage_metadata)
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
            _set_interaction_usage(response, getattr(interaction, "usage", None))
            yield interaction.output_text or ""
            return

        usage = None
        for event in interaction:
            if event.event_type == "step.delta" and event.delta.type == "text":
                yield event.delta.text
            elif event.event_type == "step.stop":
                step_usage = getattr(event, "usage", None)
                if step_usage is not None:
                    usage = step_usage
            elif event.event_type == "interaction.completed":
                completed_interaction = getattr(event, "interaction", None)
                completed_usage = getattr(completed_interaction, "usage", None)
                if completed_usage is not None:
                    usage = completed_usage

        _set_interaction_usage(response, usage)

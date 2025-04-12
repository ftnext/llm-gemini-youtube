import os

import llm
from google.genai import Client, types


@llm.hookimpl
def register_models(register):
    register(GeminiYouTube("gemini-2.0-flash-yt"))
    register(GeminiYouTube("gemini-1.5-pro-yt"))


class GeminiYouTube(llm.KeyModel):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = True
    attachment_types = set(["text/html; charset=utf-8"])  # for YouTube URLs

    def __init__(self, model_id: str):
        self.model_id = model_id

    def execute(self, prompt, stream, response, conversation, key):
        if not prompt.attachments:
            raise llm.ModelError("Attachment (YouTube URL) is required.")

        client = Client(api_key=os.getenv("LLM_GEMINI_KEY"))

        youtube_uri = None
        for attachment in prompt.attachments:
            if attachment.url and (
                "youtube.com/watch?v=" in attachment.url
                or "youtu.be/" in attachment.url
            ):
                youtube_uri = attachment.url
                break
        if not youtube_uri:
            raise llm.ModelError("YouTube URL attachment is required.")

        streaming = client.models.generate_content_stream(
            model=f"models/{self.model_id.removesuffix('-yt')}",
            contents=types.Content(
                parts=[
                    types.Part(text=prompt.prompt),
                    types.Part(file_data=types.FileData(file_uri=youtube_uri)),
                ]
            ),
        )
        for chunk in streaming:
            yield chunk.text

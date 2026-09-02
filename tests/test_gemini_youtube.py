import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import llm
import pytest
from llm.plugins import load_plugins, pm

from llm_gemini_youtube import GeminiYouTube, is_youtube_uri


def test_plugin_is_installed():
    load_plugins()

    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_gemini_youtube" in names


class TestIsYouTubeUri:
    @pytest.mark.parametrize(
        "uri",
        [
            "https://www.youtube.com/watch?v=9hE5-98ZeCg",
            "https://youtu.be/9hE5-98ZeCg",
            "https://www.youtube.com/shorts/46ycw2pQJCA",
        ],
    )
    def test_youtube_uri(self, uri):
        assert is_youtube_uri(uri)

    def test_not_youtube_uri(self):
        assert not is_youtube_uri("https://example.com")


class TestSupportedModels:
    @pytest.mark.parametrize(
        "expected_model",
        [
            "gemini-3.7-flash-yt",
            "gemini-3.6-flash-yt",
            "gemini-3.5-flash-lite-yt",
        ],
    )
    def test_contains_llm_models_output(self, expected_model):
        result = subprocess.run(
            [sys.executable, "-m", "llm", "models", "-q", "-yt"],
            check=True,
            capture_output=True,
            text=True,
        )

        assert expected_model in result.stdout


class TestExecute:
    @pytest.fixture
    def prompt(self):
        return SimpleNamespace(
            prompt="What happens in this video?",
            attachments=[
                llm.Attachment(url="https://youtu.be/7Z5Vy9JBANs"),
            ],
        )

    @patch("llm_gemini_youtube.Client")
    def test_non_streaming_interaction(self, client_class, prompt, monkeypatch):
        client = client_class.return_value
        client.interactions.create.return_value = SimpleNamespace(
            output_text="Three announcements."
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "original-key")

        def assert_environment_then_create_client():
            assert os.environ["GOOGLE_API_KEY"] == "test-key"
            return client

        client_class.side_effect = assert_environment_then_create_client
        model = GeminiYouTube("gemini-3.7-flash-yt")

        chunks = list(model.execute(prompt, False, None, None, "test-key"))

        assert chunks == ["Three announcements."]
        assert os.environ["GOOGLE_API_KEY"] == "original-key"
        client_class.assert_called_once_with()
        client.interactions.create.assert_called_once_with(
            model="gemini-3.7-flash",
            input=[
                {
                    "type": "video",
                    "uri": "https://youtu.be/7Z5Vy9JBANs",
                    "processing": "agentic",
                },
                {"type": "text", "text": "What happens in this video?"},
            ],
            stream=False,
        )

    @patch("llm_gemini_youtube.Client")
    def test_enterprise_environment_is_available_to_client(
        self, client_class, prompt, monkeypatch
    ):
        client = client_class.return_value
        client.interactions.create.return_value = SimpleNamespace(output_text="Done")
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "1")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        def assert_environment_then_create_client():
            assert os.environ["GOOGLE_API_KEY"] == "test-key"
            assert os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] == "1"
            assert os.environ["GOOGLE_CLOUD_PROJECT"] == "test-project"
            assert os.environ["GOOGLE_CLOUD_LOCATION"] == "us-central1"
            return client

        client_class.side_effect = assert_environment_then_create_client
        model = GeminiYouTube("gemini-3.7-flash-yt")

        chunks = list(model.execute(prompt, False, None, None, "test-key"))

        assert chunks == ["Done"]
        client_class.assert_called_once_with()

    @patch("llm_gemini_youtube.Client")
    def test_streaming_interaction_yields_only_text_deltas(
        self, client_class, prompt
    ):
        text_delta = SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="text", text="Hello"),
        )
        tool_delta = SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="video"),
        )
        completed = SimpleNamespace(event_type="interaction.completed")
        client_class.return_value.interactions.create.return_value = iter(
            [text_delta, tool_delta, completed]
        )
        model = GeminiYouTube("gemini-3.7-flash-yt")

        chunks = list(model.execute(prompt, True, None, None, "test-key"))

        assert chunks == ["Hello"]

    @pytest.mark.parametrize(
        "attachments",
        [
            [],
            [llm.Attachment(url="https://example.com")],
            [SimpleNamespace(url=None)],
        ],
    )
    def test_youtube_attachment_is_required(self, attachments):
        model = GeminiYouTube("gemini-3.7-flash-yt")
        prompt = SimpleNamespace(prompt="Summarize", attachments=attachments)

        with pytest.raises(llm.ModelError):
            list(model.execute(prompt, False, None, None, "test-key"))

import logging
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import llm
import pytest
import google.genai.interactions as gi
from google.genai import types
from llm.plugins import load_plugins, pm

from llm_gemini_youtube import GeminiYouTube, is_google_cloud_enabled, is_youtube_uri


def test_google_genai_models_warnings_are_suppressed():
    assert logging.getLogger("google_genai.models").level == logging.ERROR


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
            "gemini-4-carbon-yt",
            "gemini-4-neon-yt",
            "gemini-4-argon-yt",
            "gemini-4-hydrogen-yt",
            "gemini-3.8-flash-yt",
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


class TestGoogleCloudConfiguration:
    @pytest.mark.parametrize(
        ("environment_variable", "value"),
        [
            ("GOOGLE_GENAI_USE_ENTERPRISE", "1"),
            ("GOOGLE_GENAI_USE_ENTERPRISE", "true"),
            ("GOOGLE_GENAI_USE_VERTEXAI", "1"),
            ("GOOGLE_GENAI_USE_VERTEXAI", "true"),
        ],
    )
    def test_google_cloud_is_enabled(
        self, environment_variable, value, monkeypatch
    ):
        monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        monkeypatch.setenv(environment_variable, value)

        assert is_google_cloud_enabled()

    def test_enterprise_setting_takes_precedence_over_vertexai(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "false")
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")

        assert not is_google_cloud_enabled()

    def test_google_cloud_does_not_require_llm_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "1")
        model = GeminiYouTube("gemini-3.7-flash-yt")

        with patch("llm.get_key") as get_key:
            assert model.get_key() is None

        get_key.assert_not_called()

    def test_developer_api_still_uses_llm_key_lookup(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        model = GeminiYouTube("gemini-3.7-flash-yt")

        with patch("llm.get_key", return_value="test-key") as get_key:
            assert model.get_key() == "test-key"

        get_key.assert_called_once_with(
            explicit_key=None,
            key_alias="gemini",
            env_var="LLM_GEMINI_KEY",
        )


class TestExecute:
    @pytest.fixture(autouse=True)
    def clear_google_cloud_environment(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

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
    def test_enterprise_uses_generate_content(
        self, client_class, prompt, monkeypatch
    ):
        client = client_class.return_value
        client.models.generate_content.return_value = SimpleNamespace(text="Done")
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "1")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        def assert_environment_then_create_client():
            assert "GOOGLE_API_KEY" not in os.environ
            assert os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] == "1"
            assert os.environ["GOOGLE_CLOUD_PROJECT"] == "test-project"
            assert os.environ["GOOGLE_CLOUD_LOCATION"] == "global"
            return client

        client_class.side_effect = assert_environment_then_create_client
        model = GeminiYouTube("gemini-3.7-flash-yt")

        chunks = list(model.execute(prompt, False, None, None, None))

        assert chunks == ["Done"]
        client_class.assert_called_once_with()
        client.models.generate_content.assert_called_once()
        client.interactions.create.assert_not_called()

        call = client.models.generate_content.call_args
        assert call.kwargs["model"] == "gemini-3.7-flash"
        video_part, text = call.kwargs["contents"]
        assert video_part.file_data.file_uri == "https://youtu.be/7Z5Vy9JBANs"
        assert video_part.file_data.mime_type == "video/mp4"
        assert video_part.media_processing == "AGENTIC"
        assert text == "What happens in this video?"

    @patch("llm_gemini_youtube.Client")
    def test_streaming_enterprise_uses_generate_content_stream(
        self, client_class, prompt, monkeypatch
    ):
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
        client_class.return_value.models.generate_content_stream.return_value = iter(
            [
                SimpleNamespace(text="Hello"),
                SimpleNamespace(text=None),
                SimpleNamespace(text=" world"),
            ]
        )
        model = GeminiYouTube("gemini-3.7-flash-yt")

        chunks = list(model.execute(prompt, True, None, None, None))

        assert chunks == ["Hello", " world"]
        client_class.return_value.models.generate_content_stream.assert_called_once()
        client_class.return_value.interactions.create.assert_not_called()

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


class TestUsage:
    @pytest.fixture(autouse=True)
    def clear_google_cloud_environment(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    @pytest.fixture
    def model(self):
        return GeminiYouTube("gemini-3.7-flash-yt")

    @pytest.fixture
    def make_response(self, model):
        def _make_response(stream: bool = False):
            prompt = llm.Prompt(
                "What happens in this video?",
                model=model,
                attachments=[llm.Attachment(url="https://youtu.be/7Z5Vy9JBANs")],
            )
            return llm.Response(prompt, model=model, stream=stream)

        return _make_response

    @patch("llm_gemini_youtube.Client")
    def test_non_streaming_interaction_records_usage_with_thoughts_and_modality(
        self, client_class, model, make_response
    ):
        response = make_response(stream=False)
        client_class.return_value.interactions.create.return_value = SimpleNamespace(
            output_text="Done.",
            usage=gi.Usage(
                total_input_tokens=11,
                total_output_tokens=160,
                total_thought_tokens=382,
                total_cached_tokens=0,
                total_tool_use_tokens=0,
                total_tokens=553,
                input_tokens_by_modality=[
                    gi.ModalityTokens(modality="text", tokens=11)
                ],
            ),
        )

        chunks = list(
            model.execute(response.prompt, False, response, None, "test-key")
        )

        assert chunks == ["Done."]
        assert response.input_tokens == 11
        assert response.output_tokens == 542
        assert response.token_details == {
            "total_output_tokens": 160,
            "total_thought_tokens": 382,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 11}],
        }
        assert "11 input, 542 output" in response.token_usage()

    @patch("llm_gemini_youtube.Client")
    def test_non_streaming_interaction_simple_usage_sets_details_none(
        self, client_class, model, make_response
    ):
        response = make_response(stream=False)
        client_class.return_value.interactions.create.return_value = SimpleNamespace(
            output_text="Done.",
            usage=gi.Usage(
                total_input_tokens=50,
                total_output_tokens=10,
                total_tokens=60,
            ),
        )

        list(model.execute(response.prompt, False, response, None, "test-key"))

        assert response.input_tokens == 50
        assert response.output_tokens == 10
        assert response.token_details is None
        assert response.token_usage() == "50 input, 10 output"

    @patch("llm_gemini_youtube.Client")
    def test_non_streaming_interaction_combines_tool_use_and_keeps_cached(
        self, client_class, model, make_response
    ):
        response = make_response(stream=False)
        client_class.return_value.interactions.create.return_value = SimpleNamespace(
            output_text="Done.",
            usage=gi.Usage(
                total_input_tokens=100,
                total_tool_use_tokens=30,
                total_cached_tokens=40,
                total_output_tokens=20,
                total_tokens=150,
            ),
        )

        list(model.execute(response.prompt, False, response, None, "test-key"))

        # Cached tokens are already part of total_input_tokens, so only tool use is added.
        assert response.input_tokens == 130
        assert response.output_tokens == 20
        assert response.token_details == {
            "total_input_tokens": 100,
            "total_tool_use_tokens": 30,
            "total_cached_tokens": 40,
        }

    @patch("llm_gemini_youtube.Client")
    def test_streaming_interaction_prefers_completed_event_usage(
        self, client_class, model, make_response
    ):
        response = make_response(stream=True)
        events = [
            SimpleNamespace(
                event_type="step.delta",
                delta=SimpleNamespace(type="text", text="Hello"),
            ),
            SimpleNamespace(
                event_type="step.stop",
                usage=gi.Usage(total_input_tokens=10, total_output_tokens=2),
            ),
            SimpleNamespace(
                event_type="interaction.completed",
                interaction=SimpleNamespace(
                    usage=gi.Usage(
                        total_input_tokens=25,
                        total_output_tokens=8,
                        total_thought_tokens=4,
                        total_tokens=37,
                    )
                ),
            ),
        ]
        client_class.return_value.interactions.create.return_value = iter(events)

        chunks = list(
            model.execute(response.prompt, True, response, None, "test-key")
        )

        assert chunks == ["Hello"]
        assert response.input_tokens == 25
        assert response.output_tokens == 12
        assert response.token_details == {
            "total_output_tokens": 8,
            "total_thought_tokens": 4,
        }

    @patch("llm_gemini_youtube.Client")
    def test_streaming_interaction_falls_back_to_last_step_stop_usage(
        self, client_class, model, make_response
    ):
        response = make_response(stream=True)
        events = [
            SimpleNamespace(
                event_type="step.delta",
                delta=SimpleNamespace(type="text", text="Hello"),
            ),
            SimpleNamespace(
                event_type="step.stop",
                usage=gi.Usage(total_input_tokens=10, total_output_tokens=2),
            ),
            SimpleNamespace(
                event_type="step.stop",
                usage=gi.Usage(total_input_tokens=30, total_output_tokens=7),
            ),
            SimpleNamespace(
                event_type="interaction.completed",
                interaction=SimpleNamespace(usage=None),
            ),
        ]
        client_class.return_value.interactions.create.return_value = iter(events)

        chunks = list(
            model.execute(response.prompt, True, response, None, "test-key")
        )

        assert chunks == ["Hello"]
        assert response.input_tokens == 30
        assert response.output_tokens == 7
        assert response.token_details is None

    @patch("llm_gemini_youtube.Client")
    def test_enterprise_non_streaming_records_usage_metadata(
        self, client_class, model, make_response, monkeypatch
    ):
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "1")
        response = make_response(stream=False)
        usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=8974,
            candidates_token_count=972,
            thoughts_token_count=1658,
            total_token_count=11604,
            prompt_tokens_details=[
                types.ModalityTokenCount(
                    modality=types.MediaModality.VIDEO, token_count=6468
                ),
                types.ModalityTokenCount(
                    modality=types.MediaModality.TEXT, token_count=56
                ),
                types.ModalityTokenCount(
                    modality=types.MediaModality.AUDIO, token_count=2450
                ),
            ],
            candidates_tokens_details=[
                types.ModalityTokenCount(
                    modality=types.MediaModality.TEXT, token_count=972
                )
            ],
            traffic_type=types.TrafficType.ON_DEMAND,
        )
        client_class.return_value.models.generate_content.return_value = (
            SimpleNamespace(text="Done", usage_metadata=usage_metadata)
        )

        chunks = list(model.execute(response.prompt, False, response, None, None))

        assert chunks == ["Done"]
        assert response.input_tokens == 8974
        assert response.output_tokens == 2630
        assert response.token_details == {
            "candidates_token_count": 972,
            "thoughts_token_count": 1658,
            "prompt_tokens_details": [
                {"modality": "VIDEO", "token_count": 6468},
                {"modality": "TEXT", "token_count": 56},
                {"modality": "AUDIO", "token_count": 2450},
            ],
            "candidates_tokens_details": [
                {"modality": "TEXT", "token_count": 972},
            ],
            "traffic_type": "ON_DEMAND",
        }
        assert "8,974 input, 2,630 output" in response.token_usage()

    @patch("llm_gemini_youtube.Client")
    def test_enterprise_non_streaming_combines_tool_use_and_keeps_cached(
        self, client_class, model, make_response, monkeypatch
    ):
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "1")
        response = make_response(stream=False)
        usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=200,
            tool_use_prompt_token_count=45,
            cached_content_token_count=80,
            candidates_token_count=30,
            total_token_count=275,
        )
        client_class.return_value.models.generate_content.return_value = (
            SimpleNamespace(text="Done", usage_metadata=usage_metadata)
        )

        list(model.execute(response.prompt, False, response, None, None))

        assert response.input_tokens == 245
        assert response.output_tokens == 30
        assert response.token_details == {
            "prompt_token_count": 200,
            "tool_use_prompt_token_count": 45,
            "cached_content_token_count": 80,
        }

    @patch("llm_gemini_youtube.Client")
    def test_enterprise_streaming_records_final_chunk_usage_metadata(
        self, client_class, model, make_response, monkeypatch
    ):
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
        response = make_response(stream=True)
        chunks_stream = [
            SimpleNamespace(
                text="Hello",
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=100,
                    candidates_token_count=2,
                    total_token_count=102,
                ),
            ),
            SimpleNamespace(text=" world", usage_metadata=None),
            SimpleNamespace(
                text=None,
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=100,
                    candidates_token_count=12,
                    thoughts_token_count=5,
                    total_token_count=117,
                ),
            ),
        ]
        client_class.return_value.models.generate_content_stream.return_value = iter(
            chunks_stream
        )

        chunks = list(model.execute(response.prompt, True, response, None, None))

        assert chunks == ["Hello", " world"]
        assert response.input_tokens == 100
        assert response.output_tokens == 17
        assert response.token_details == {
            "candidates_token_count": 12,
            "thoughts_token_count": 5,
        }

    @pytest.mark.parametrize(
        "empty_usage",
        [
            None,
            gi.Usage(),
            gi.Usage(total_cached_tokens=0, total_tool_use_tokens=0),
        ],
    )
    @patch("llm_gemini_youtube.Client")
    def test_empty_interaction_usage_does_not_call_set_usage(
        self, client_class, model, make_response, empty_usage
    ):
        response = make_response(stream=False)
        client_class.return_value.interactions.create.return_value = SimpleNamespace(
            output_text="Done.",
            usage=empty_usage,
        )

        with patch.object(
            response, "set_usage", wraps=response.set_usage
        ) as set_usage_spy:
            list(model.execute(response.prompt, False, response, None, "test-key"))

        set_usage_spy.assert_not_called()
        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.token_details is None

    @patch("llm_gemini_youtube.Client")
    def test_empty_generate_content_usage_does_not_call_set_usage(
        self, client_class, model, make_response, monkeypatch
    ):
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "1")
        response = make_response(stream=False)
        client_class.return_value.models.generate_content.return_value = (
            SimpleNamespace(
                text="Done",
                usage_metadata=types.GenerateContentResponseUsageMetadata(),
            )
        )

        with patch.object(
            response, "set_usage", wraps=response.set_usage
        ) as set_usage_spy:
            list(model.execute(response.prompt, False, response, None, None))

        set_usage_spy.assert_not_called()
        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.token_details is None

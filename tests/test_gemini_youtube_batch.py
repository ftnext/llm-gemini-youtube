import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli
from llm.parts import Message, TextPart

from llm_gemini_youtube import (
    GeminiYouTubeBatch,
    build_batch_request,
    parse_batch_input_uri,
)


JOB_NAME = "projects/123456789/locations/global/batchPredictionJobs/987654321"
INPUT_URI = "gs://my-bucket/youtube-batch/input.jsonl"
VIDEO_1 = "https://www.youtube.com/watch?v=VIDEO1"
VIDEO_2 = "https://www.youtube.com/watch?v=VIDEO2"
VIDEO_3 = "https://www.youtube.com/watch?v=VIDEO3"
PROMPT_TEXT = "要約して"


def youtube_attachment(url):
    return llm.Attachment(type="text/html; charset=utf-8", url=url)


@pytest.fixture
def google_cloud(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")


@pytest.fixture
def google_cloud_disabled(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)


@pytest.fixture
def genai_client():
    with patch("llm_gemini_youtube.Client") as client_class:
        client_class.return_value.batches.create.return_value = SimpleNamespace(
            name=JOB_NAME
        )
        yield client_class


@pytest.fixture
def storage_client():
    with patch("llm_gemini_youtube.storage.Client") as client_class:
        yield client_class


def batch_prompt(model, prompt_text=PROMPT_TEXT, urls=(VIDEO_1,), options=None):
    return model.prompt(
        prompt_text,
        attachments=[youtube_attachment(url) for url in urls],
        options=options if options is not None else {"batch_input_uri": INPUT_URI},
    )


def uploaded_jsonl(storage_client_class):
    blob = (
        storage_client_class.return_value.bucket.return_value.blob.return_value
    )
    return blob.upload_from_string.call_args.args[0]


class TestBatchCliModels:
    @pytest.mark.parametrize(
        "expected_model",
        [
            "gemini-3.8-flash-yt-batch",
            "gemini-3.7-flash-yt-batch",
            "gemini-3.6-flash-yt-batch",
            "gemini-3.5-flash-lite-yt-batch",
        ],
    )
    def test_contains_llm_models_output(self, expected_model):
        result = subprocess.run(
            [sys.executable, "-m", "llm", "models", "-q", "yt-batch"],
            check=True,
            capture_output=True,
            text=True,
        )

        assert expected_model in result.stdout


class TestParseBatchInputUri:
    def test_parses_bucket_and_object(self):
        assert parse_batch_input_uri(INPUT_URI) == (
            "my-bucket",
            "youtube-batch/input.jsonl",
        )

    @pytest.mark.parametrize(
        "uri",
        [
            "https://example.com/input.jsonl",
            "gs://my-bucket",
            "gs://my-bucket/",
            "gs://my-bucket/input",
            "gs://my-bucket/input.json",
        ],
    )
    def test_rejects_invalid_uri(self, uri):
        with pytest.raises(llm.ModelError):
            parse_batch_input_uri(uri)


class TestBuildBatchRequest:
    def test_structure(self):
        request = build_batch_request(VIDEO_1, "要約して")
        assert request == {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "fileData": {
                                    "fileUri": VIDEO_1,
                                    "mimeType": "video/mp4",
                                },
                                "mediaProcessing": "AGENTIC",
                            },
                            {"text": "要約して"},
                        ],
                    }
                ]
            }
        }


class TestBatchExecute:
    def test_one_job_for_three_videos(
        self, google_cloud, genai_client, storage_client
    ):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        prompt = batch_prompt(model, urls=(VIDEO_1, VIDEO_2, VIDEO_3))

        chunks = list(model.execute(prompt.prompt, False, None, None, None))

        assert chunks == [JOB_NAME]
        genai_client.assert_called_once()
        assert (
            genai_client.call_args.kwargs["http_options"].api_version == "v1beta1"
        )
        genai_client.return_value.batches.create.assert_called_once_with(
            model="gemini-3.8-flash", src=INPUT_URI
        )
        storage_client.return_value.bucket.assert_called_once_with("my-bucket")
        storage_client.return_value.bucket.return_value.blob.assert_called_once_with(
            "youtube-batch/input.jsonl"
        )

        lines = uploaded_jsonl(storage_client).split("\n")
        assert len(lines) == 3
        for line, url in zip(lines, (VIDEO_1, VIDEO_2, VIDEO_3)):
            assert json.loads(line) == build_batch_request(url, PROMPT_TEXT)

    def test_duplicate_urls_produce_duplicate_requests(
        self, google_cloud, genai_client, storage_client
    ):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        prompt = batch_prompt(model, urls=(VIDEO_1, VIDEO_1))

        list(model.execute(prompt.prompt, False, None, None, None))

        lines = uploaded_jsonl(storage_client).split("\n")
        assert len(lines) == 2
        assert lines[0] == lines[1]
        genai_client.return_value.batches.create.assert_called_once()

    def test_single_video_is_allowed(
        self, google_cloud, genai_client, storage_client
    ):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        prompt = batch_prompt(model, urls=(VIDEO_1,))

        list(model.execute(prompt.prompt, False, None, None, None))

        assert len(uploaded_jsonl(storage_client).split("\n")) == 1

    def test_same_input_uri_is_overwritten(
        self, google_cloud, genai_client, storage_client
    ):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")

        list(model.execute(batch_prompt(model).prompt, False, None, None, None))
        list(model.execute(batch_prompt(model).prompt, False, None, None, None))

        blob = (
            storage_client.return_value.bucket.return_value.blob.return_value
        )
        assert blob.upload_from_string.call_count == 2

    def test_upload_failure_skips_job_creation(
        self, google_cloud, genai_client, storage_client
    ):
        storage_client.return_value.bucket.return_value.blob.return_value.upload_from_string.side_effect = RuntimeError(
            "upload failed"
        )
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")

        with pytest.raises(RuntimeError):
            list(model.execute(batch_prompt(model).prompt, False, None, None, None))

        genai_client.return_value.batches.create.assert_not_called()

    def test_job_creation_failure_is_an_error(
        self, google_cloud, genai_client, storage_client
    ):
        genai_client.return_value.batches.create.side_effect = RuntimeError(
            "create failed"
        )
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")

        with pytest.raises(RuntimeError):
            list(model.execute(batch_prompt(model).prompt, False, None, None, None))


class TestBatchValidation:
    def test_google_cloud_is_required(
        self, google_cloud_disabled, genai_client, storage_client
    ):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")

        with pytest.raises(llm.ModelError, match="requires Google Cloud"):
            list(
                model.execute(batch_prompt(model).prompt, False, None, None, None)
            )

        storage_client.assert_not_called()
        genai_client.assert_not_called()

    def test_cloud_error_precedes_key_lookup(self, google_cloud_disabled):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")

        with patch("llm.get_key", return_value="registered-key") as get_key:
            with pytest.raises(llm.ModelError, match="requires Google Cloud"):
                model.get_key()

        get_key.assert_not_called()

    def test_cloud_error_precedes_key_lookup_with_explicit_key(
        self, google_cloud_disabled
    ):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")

        with patch("llm.get_key") as get_key:
            with pytest.raises(llm.ModelError, match="requires Google Cloud"):
                model.get_key("explicit-key")

        get_key.assert_not_called()

    def test_system_prompt_is_rejected(self, google_cloud):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        response = model.prompt(
            PROMPT_TEXT,
            attachments=[youtube_attachment(VIDEO_1)],
            system="be terse",
            options={"batch_input_uri": INPUT_URI},
        )

        with pytest.raises(llm.ModelError, match="system prompt"):
            list(model.execute(response.prompt, False, None, None, None))

    def test_no_attachments_is_an_error(self, google_cloud):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        response = model.prompt(
            PROMPT_TEXT, options={"batch_input_uri": INPUT_URI}
        )

        with pytest.raises(llm.ModelError, match="Attachment"):
            list(model.execute(response.prompt, False, None, None, None))

    def test_non_youtube_attachment_is_rejected(self, google_cloud):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        response = model.prompt(
            PROMPT_TEXT,
            attachments=[
                youtube_attachment(VIDEO_1),
                llm.Attachment(
                    type="text/html; charset=utf-8", url="https://example.com"
                ),
            ],
            options={"batch_input_uri": INPUT_URI},
        )

        with pytest.raises(llm.ModelError, match="YouTube URL"):
            list(model.execute(response.prompt, False, None, None, None))

    def test_attachment_without_url_is_rejected(self, google_cloud):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        response = model.prompt(
            PROMPT_TEXT,
            attachments=[
                llm.Attachment(type="text/html; charset=utf-8", path="/tmp/x")
            ],
            options={"batch_input_uri": INPUT_URI},
        )

        with pytest.raises(llm.ModelError, match="YouTube URL"):
            list(model.execute(response.prompt, False, None, None, None))

    def test_batch_input_uri_is_required(self, google_cloud):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        response = model.prompt(
            PROMPT_TEXT, attachments=[youtube_attachment(VIDEO_1)]
        )

        with pytest.raises(llm.ModelError, match="batch_input_uri"):
            list(model.execute(response.prompt, False, None, None, None))

    def test_invalid_batch_input_uri_is_rejected(self, google_cloud):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        response = model.prompt(
            PROMPT_TEXT,
            attachments=[youtube_attachment(VIDEO_1)],
            options={"batch_input_uri": "gs://my-bucket/input.txt"},
        )

        with pytest.raises(llm.ModelError, match="batch_input_uri"):
            list(model.execute(response.prompt, False, None, None, None))


class TestBatchHistory:
    def test_conversation_prompt_is_rejected(
        self, google_cloud, genai_client, storage_client
    ):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        conversation = model.conversation()
        first = conversation.prompt(
            PROMPT_TEXT,
            attachments=[youtube_attachment(VIDEO_1)],
            options={"batch_input_uri": INPUT_URI},
        )
        list(first)

        with pytest.raises(llm.ConversationNotSupported):
            conversation.prompt(
                "next",
                attachments=[youtube_attachment(VIDEO_2)],
                options={"batch_input_uri": INPUT_URI},
            )

    def test_messages_history_is_rejected(
        self, google_cloud, genai_client, storage_client
    ):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        response = model.prompt(
            PROMPT_TEXT,
            attachments=[youtube_attachment(VIDEO_1)],
            messages=[
                Message(role="user", parts=[TextPart(text="earlier turn")])
            ],
            options={"batch_input_uri": INPUT_URI},
        )

        with pytest.raises(llm.ModelError, match="history"):
            list(model.execute(response.prompt, False, None, None, None))

        genai_client.return_value.batches.create.assert_not_called()
        storage_client.assert_not_called()

    def test_reply_is_rejected(
        self, google_cloud, genai_client, storage_client
    ):
        model = GeminiYouTubeBatch("gemini-3.8-flash-yt-batch")
        response = model.prompt(
            PROMPT_TEXT,
            attachments=[youtube_attachment(VIDEO_1)],
            options={"batch_input_uri": INPUT_URI},
        )
        list(response)

        with pytest.raises(llm.ConversationNotSupported):
            response.reply(
                "next",
                attachments=[youtube_attachment(VIDEO_2)],
                options={"batch_input_uri": INPUT_URI},
            )


def head_response(content_type="text/html; charset=utf-8"):
    return SimpleNamespace(
        headers={"content-type": content_type},
        raise_for_status=lambda: None,
    )


@pytest.fixture
def head_ok():
    with patch("httpx2.head", return_value=head_response()) as head:
        yield head


class TestBatchCli:
    @pytest.fixture(autouse=True)
    def _disable_llm_key_lookup(self):
        with patch("llm.get_key", return_value=None) as get_key:
            self.get_key = get_key
            yield

    def invoke(self, args, **kwargs):
        runner = CliRunner()
        return runner.invoke(cli, args, **kwargs)

    def test_cli_submits_one_batch_job(
        self, google_cloud, head_ok, genai_client, storage_client, tmp_path
    ):
        result = self.invoke(
            [
                "-m",
                "gemini-3.8-flash-yt-batch",
                "-o",
                "batch_input_uri",
                INPUT_URI,
                "-d",
                str(tmp_path / "logs.db"),
                "-a",
                VIDEO_1,
                "-a",
                VIDEO_2,
                "-a",
                VIDEO_3,
                PROMPT_TEXT,
            ]
        )

        assert result.exit_code == 0, result.output
        assert result.output.strip() == JOB_NAME
        genai_client.return_value.batches.create.assert_called_once_with(
            model="gemini-3.8-flash", src=INPUT_URI
        )
        assert len(uploaded_jsonl(storage_client).split("\n")) == 3

    def test_cli_no_stream_prints_job_name(
        self, google_cloud, head_ok, genai_client, storage_client, tmp_path
    ):
        result = self.invoke(
            [
                "-m",
                "gemini-3.8-flash-yt-batch",
                "--no-stream",
                "-o",
                "batch_input_uri",
                INPUT_URI,
                "-d",
                str(tmp_path / "logs.db"),
                "-a",
                VIDEO_1,
                PROMPT_TEXT,
            ]
        )

        assert result.exit_code == 0, result.output
        assert result.output.strip() == JOB_NAME

    def test_cli_json_output_contains_job_name(
        self, google_cloud, head_ok, genai_client, storage_client, tmp_path
    ):
        result = self.invoke(
            [
                "-m",
                "gemini-3.8-flash-yt-batch",
                "--json",
                "-o",
                "batch_input_uri",
                INPUT_URI,
                "-d",
                str(tmp_path / "logs.db"),
                "-a",
                VIDEO_1,
                PROMPT_TEXT,
            ]
        )

        assert result.exit_code == 0, result.output
        rows = json.loads(result.output)
        assert JOB_NAME in json.dumps(rows)

    def test_cli_head_failure_blocks_submission(
        self, google_cloud, genai_client, storage_client, tmp_path
    ):
        with patch(
            "httpx2.head", side_effect=httpx2.HTTPError("connection failed")
        ):
            result = self.invoke(
                [
                    "-m",
                    "gemini-3.8-flash-yt-batch",
                    "-o",
                    "batch_input_uri",
                    INPUT_URI,
                    "-d",
                    str(tmp_path / "logs.db"),
                    "-a",
                    VIDEO_1,
                    PROMPT_TEXT,
                ]
            )

        assert result.exit_code != 0
        genai_client.return_value.batches.create.assert_not_called()
        storage_client.assert_not_called()

    def test_cli_mime_mismatch_blocks_submission(
        self, google_cloud, genai_client, storage_client, tmp_path
    ):
        with patch(
            "httpx2.head",
            return_value=head_response(content_type="text/html"),
        ):
            result = self.invoke(
                [
                    "-m",
                    "gemini-3.8-flash-yt-batch",
                    "-o",
                    "batch_input_uri",
                    INPUT_URI,
                    "-d",
                    str(tmp_path / "logs.db"),
                    "-a",
                    VIDEO_1,
                    PROMPT_TEXT,
                ]
            )

        assert result.exit_code != 0
        assert "attachment" in result.output
        genai_client.return_value.batches.create.assert_not_called()
        storage_client.assert_not_called()

    def test_cli_non_youtube_url_blocks_submission(
        self, google_cloud, head_ok, genai_client, storage_client, tmp_path
    ):
        result = self.invoke(
            [
                "-m",
                "gemini-3.8-flash-yt-batch",
                "-o",
                "batch_input_uri",
                INPUT_URI,
                "-d",
                str(tmp_path / "logs.db"),
                "-a",
                VIDEO_1,
                "-a",
                "https://example.com",
                PROMPT_TEXT,
            ]
        )

        assert result.exit_code != 0
        assert "YouTube URL" in result.output
        genai_client.return_value.batches.create.assert_not_called()
        storage_client.assert_not_called()

    def test_cli_requires_google_cloud(
        self, google_cloud_disabled, head_ok, genai_client, storage_client, tmp_path
    ):
        result = self.invoke(
            [
                "-m",
                "gemini-3.8-flash-yt-batch",
                "-o",
                "batch_input_uri",
                INPUT_URI,
                "-d",
                str(tmp_path / "logs.db"),
                "-a",
                VIDEO_1,
                PROMPT_TEXT,
            ]
        )

        assert result.exit_code != 0
        assert "Google Cloud" in result.output
        assert "llm keys set" not in result.output
        self.get_key.assert_not_called()
        genai_client.return_value.batches.create.assert_not_called()
        storage_client.assert_not_called()

    def test_cli_requires_batch_input_uri(
        self, google_cloud, head_ok, genai_client, storage_client, tmp_path
    ):
        result = self.invoke(
            [
                "-m",
                "gemini-3.8-flash-yt-batch",
                "-d",
                str(tmp_path / "logs.db"),
                "-a",
                VIDEO_1,
                PROMPT_TEXT,
            ]
        )

        assert result.exit_code != 0
        assert "batch_input_uri" in result.output
        genai_client.return_value.batches.create.assert_not_called()
        storage_client.assert_not_called()

    def test_cli_system_prompt_is_rejected(
        self, google_cloud, head_ok, genai_client, storage_client, tmp_path
    ):
        result = self.invoke(
            [
                "-m",
                "gemini-3.8-flash-yt-batch",
                "-s",
                "be terse",
                "-o",
                "batch_input_uri",
                INPUT_URI,
                "-d",
                str(tmp_path / "logs.db"),
                "-a",
                VIDEO_1,
                PROMPT_TEXT,
            ]
        )

        assert result.exit_code != 0
        assert "system prompt" in result.output
        storage_client.assert_not_called()

    def test_cli_continue_is_rejected(
        self, google_cloud, head_ok, genai_client, storage_client, tmp_path
    ):
        db_path = str(tmp_path / "logs.db")
        first = self.invoke(
            [
                "-m",
                "gemini-3.8-flash-yt-batch",
                "-o",
                "batch_input_uri",
                INPUT_URI,
                "-d",
                db_path,
                "-a",
                VIDEO_1,
                PROMPT_TEXT,
            ]
        )
        assert first.exit_code == 0, first.output

        second = self.invoke(
            [
                "-c",
                "-o",
                "batch_input_uri",
                INPUT_URI,
                "-d",
                db_path,
                "-a",
                VIDEO_2,
                "next",
            ]
        )

        assert second.exit_code != 0
        assert "conversation" in second.output
        assert genai_client.return_value.batches.create.call_count == 1

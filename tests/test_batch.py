import json
from unittest.mock import patch

import llm
import pytest
from click.testing import CliRunner
from google.api_core.exceptions import Forbidden
from google.genai import types
from google.genai.errors import ClientError
from llm.cli import cli

from llm_gemini_youtube import GeminiVideoBatch, batch_jsonl


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("LLM_GEMINI_BATCH_INPUT_PREFIX", "gs://inputs/requests")
    monkeypatch.setenv("LLM_GEMINI_BATCH_OUTPUT_PREFIX", "gs://outputs/results")


@pytest.fixture
def services(cloud):
    with patch("llm_gemini_youtube.Client") as genai, patch(
        "llm_gemini_youtube.storage.Client"
    ) as storage:
        client = genai.return_value.__enter__.return_value
        uploader = storage.return_value.__enter__.return_value
        client.batches.create.return_value = types.BatchJob(
            name="projects/test-project/locations/global/batchPredictionJobs/123",
            state=types.JobState.JOB_STATE_PENDING,
        )
        yield client, uploader


def test_jsonl_preserves_all_videos_and_prompt():
    videos = ["https://youtu.be/first", "https://youtu.be/example", "https://www.youtube.com/watch?v=second"]
    payload = batch_jsonl(videos, '要約して\n"詳細も"')
    rows = [json.loads(line) for line in payload.splitlines()]
    assert len(rows) == 3
    for uri, row in zip(videos, rows):
        assert row == {"request": {"contents": [{"role": "user", "parts": [
            {"fileData": {"fileUri": uri, "mimeType": "video/mp4"}, "mediaProcessing": "AGENTIC"},
            {"text": '要約して\n"詳細も"'},
        ]}]}}
    assert payload.endswith("\n")


@pytest.mark.parametrize("videos", [[], ["/tmp/video.mp4"], ["https://example.com/a.mp4"], ["gs://videos/a.txt"], ["gs:///a.mp4"], ["gs://videos/a.mp4"]])
def test_invalid_videos(videos):
    with pytest.raises(llm.ModelError):
        batch_jsonl(videos, "Summarize")


def test_submit_one_job_and_receipt_without_polling(services):
    client, storage = services
    model = llm.get_model("gemini-3.8-flash-batch")
    assert isinstance(model, GeminiVideoBatch)
    assert not model.can_stream
    response = model.prompt("Summarize", videos=["https://youtu.be/first", "https://youtu.be/example"])
    receipt = json.loads(response.text())
    assert receipt["name"].endswith("/123")
    assert receipt["state"] == "JOB_STATE_PENDING"
    assert receipt["mode"] == "batch"
    assert receipt["request_count"] == 2
    assert receipt["model"] == "gemini-3.8-flash"
    assert response.response_json == receipt
    client.batches.create.assert_called_once_with(
        model="gemini-3.8-flash", src=receipt["input_uri"],
        config=types.CreateBatchJobConfig(dest=receipt["output_uri"]),
    )
    storage.bucket.assert_called_once_with("inputs")
    upload = storage.bucket.return_value.blob.return_value.upload_from_string
    assert len(upload.call_args.args[0].splitlines()) == 2
    assert upload.call_args.kwargs == {"content_type": "application/jsonl", "if_generation_match": 0}
    client.batches.get.assert_not_called()
    client.models.generate_content.assert_not_called()
    client.interactions.create.assert_not_called()


def test_cli_json_list_and_prefix_options(services):
    client, storage = services
    result = CliRunner().invoke(cli, [
        "prompt", "Summarize", "--no-log", "-m", "gemini-3.8-flash-batch",
        "-o", "videos", '["https://youtu.be/first", "https://youtu.be/example"]',
        "-o", "input_prefix", "gs://other-input/jobs",
        "-o", "output_prefix", "gs://other-output/jobs",
    ])
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["input_uri"].startswith("gs://other-input/jobs/")
    assert receipt["output_uri"].startswith("gs://other-output/jobs/")
    assert receipt["request_count"] == 2
    storage.bucket.assert_called_once_with("other-input")
    client.batches.get.assert_not_called()


def test_cli_repeated_attachments(services):
    result = CliRunner().invoke(cli, [
        "prompt", "Summarize", "--no-log", "-m", "gemini-3.8-flash-batch",
        "--at", "https://youtu.be/example", "text/html",
        "--at", "https://youtu.be/first", "text/html",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["request_count"] == 2


@pytest.mark.parametrize("variable", ["GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "LLM_GEMINI_BATCH_INPUT_PREFIX", "LLM_GEMINI_BATCH_OUTPUT_PREFIX"])
def test_missing_configuration_fails_before_upload(services, monkeypatch, variable):
    monkeypatch.delenv(variable)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    with pytest.raises(llm.ModelError):
        GeminiVideoBatch("gemini-3.8-flash-batch").prompt("Summarize", videos=["https://youtu.be/first"]).text()
    services[1].bucket.assert_not_called()
    services[0].batches.create.assert_not_called()


def test_upload_failure_does_not_create_job(services):
    client, storage = services
    storage.bucket.return_value.blob.return_value.upload_from_string.side_effect = Forbidden("denied")
    with pytest.raises(llm.ModelError, match="upload input failed"):
        GeminiVideoBatch("gemini-3.8-flash-batch").prompt("Summarize", videos=["https://youtu.be/first"]).text()
    client.batches.create.assert_not_called()


def test_submission_error_is_actionable_and_not_retried(services):
    client, _ = services
    client.batches.create.side_effect = ClientError(400, {"error": {"message": "invalid"}})
    with pytest.raises(llm.ModelError, match="check Cloud Console before retrying"):
        GeminiVideoBatch("gemini-3.8-flash-batch").prompt("Summarize", videos=["https://youtu.be/first"]).text()
    client.batches.create.assert_called_once()
    client.batches.get.assert_not_called()


def test_empty_input_does_not_upload(services):
    with pytest.raises(llm.ModelError, match="At least one video"):
        GeminiVideoBatch("gemini-3.8-flash-batch").prompt("Summarize").text()
    services[1].bucket.assert_not_called()


def test_system_prompt_is_not_silently_discarded(services):
    with pytest.raises(llm.ModelError, match="system prompts"):
        GeminiVideoBatch("gemini-3.8-flash-batch").prompt(
            "Summarize", system="In Japanese", videos=["https://youtu.be/example"]
        ).text()
    services[1].bucket.assert_not_called()


def test_each_submission_has_unique_artifact_paths(services):
    model = GeminiVideoBatch("gemini-3.8-flash-batch")
    receipts = [json.loads(model.prompt("Summarize", videos=["https://youtu.be/example"]).text()) for _ in range(2)]
    assert receipts[0]["input_uri"] != receipts[1]["input_uri"]
    assert receipts[0]["output_uri"] != receipts[1]["output_uri"]

# llm-gemini-youtube

[![PyPI](https://img.shields.io/pypi/v/llm-gemini-youtube.svg)](https://pypi.org/project/llm-gemini-youtube/)
[![Changelog](https://img.shields.io/github/v/release/ftnext/llm-gemini-youtube?include_prereleases&label=changelog)](https://github.com/ftnext/llm-gemini-youtube/releases)
[![Tests](https://github.com/ftnext/llm-gemini-youtube/actions/workflows/test.yml/badge.svg)](https://github.com/ftnext/llm-gemini-youtube/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ftnext/llm-gemini-youtube/blob/main/LICENSE)

LLM plugin for [agentic YouTube video understanding with Gemini](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-agentic-video-in-gemini/).

The plugin enables agentic processing for every video. It uses the Gemini Interactions API with an API key and the GenerateContent API with Gemini Enterprise Agent Platform. Gemini can selectively inspect the frames, audio, and transcript needed to answer the prompt instead of processing the whole video at a fixed frame rate.

## Project history

- The 0.0.x series provided server-side YouTube URL processing. That capability was later adopted upstream by `simonw/llm-gemini` in [PR #112: Add support for media_resolution and processing Youtube URLs server-side](https://github.com/simonw/llm-gemini/pull/112).
- Starting with the 0.1.x series, this project was rebooted to focus exclusively on Google's [agentic video understanding with Gemini](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-agentic-video-in-gemini/).

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-gemini-youtube
```
## Usage

```bash
llm -m gemini-3.7-flash-yt -a 'https://www.youtube.com/watch?v=9hE5-98ZeCg' 'Can you summarize this video?'

llm -m gemini-3.6-flash-yt -a 'https://www.youtube.com/watch?v=9hE5-98ZeCg' 'What are the examples given at 01:05 and 01:19 supposed to show us?'

llm -m gemini-3.5-flash-lite-yt -a 'https://www.youtube.com/watch?v=9hE5-98ZeCg' 'Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions.'
```

Supported synchronous models are `gemini-3.8-flash-yt`, `gemini-3.7-flash-yt`, `gemini-3.6-flash-yt`, and `gemini-3.5-flash-lite-yt`. Configure the API key using LLM's standard key mechanism:

```bash
llm keys set gemini
```

### Google Cloud

The Google Gen AI SDK can route requests through Gemini Enterprise Agent Platform using its standard environment variables:

```bash
export GOOGLE_GENAI_USE_ENTERPRISE=1
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='global'
gcloud auth application-default login
```

When Google Cloud is enabled, `LLM_GEMINI_KEY` does not need to be set. The SDK uses Application Default Credentials together with the configured project and location. The plugin uses the GenerateContent API with `media_processing="AGENTIC"` for this configuration; API-key requests continue to use the Interactions API with `processing="agentic"`.

### Batch submission (Google Cloud only)

Use `gemini-3.8-flash-batch` to submit a list of YouTube videos as **one batch job**.
The plugin removes `-batch` before calling the API with `gemini-3.8-flash`.
This is Google Cloud batch inference, not Gemini Developer API batch processing.
The existing `-yt` models retain their synchronous behavior.

Configure Google Cloud and ADC as above, then set existing Cloud Storage locations
for the request JSONL and output. Storage is used only for these batch artifacts;
video inputs are YouTube URLs, not GCS objects or local files.

```bash
export LLM_GEMINI_BATCH_INPUT_PREFIX='gs://your-bucket/batch-inputs'
export LLM_GEMINI_BATCH_OUTPUT_PREFIX='gs://your-bucket/batch-outputs'

llm -m gemini-3.8-flash-batch \
  -o videos '["https://www.youtube.com/watch?v=9hE5-98ZeCg", "https://youtu.be/7Z5Vy9JBANs"]' \
  'Summarize this video' > batch-job.json
```

For a larger list, save a JSON array of URL strings to `videos.json` and pass
`-o videos "$(cat videos.json)"`. Repeated attachments also work; using explicit
types avoids LLM fetching the URL to determine its MIME type:

```bash
llm -m gemini-3.8-flash-batch \
  --at 'https://youtu.be/9hE5-98ZeCg' 'text/html' \
  --at 'https://youtu.be/7Z5Vy9JBANs' 'text/html' \
  'Summarize this video'
```

`-o input_prefix gs://bucket/path` and `-o output_prefix gs://bucket/path`
override the corresponding environment variables. Each submission uses a unique
input filename and output directory. When both `videos` and attachments are
provided, attachment URLs follow the list, preserving order and duplicates.
The Python API accepts a list directly:

```python
import json
import llm

response = llm.get_model("gemini-3.8-flash-batch").prompt(
    "Summarize this video",
    videos=["https://youtu.be/9hE5-98ZeCg", "https://youtu.be/7Z5Vy9JBANs"],
)
receipt = json.loads(response.text())  # Submits the job; does not wait for completion.
print(receipt["name"], receipt["state"], receipt["output_uri"])
```

The receipt contains `mode`, `name`, `state` (initial API state, possibly null),
`model`, `input_uri`, `output_uri` (requested output prefix), and `request_count`.
It is returned as JSON text and stored in LLM's `response_json`; it is **not a
video summary**. No polling, completion waiting, output downloads, or result
post-processing are performed. Each JSONL row carries one video and the same
prompt with `mediaProcessing: AGENTIC`, matching the normal Cloud request.
The normal model has no custom generation settings or structured-output support
to inherit. Batch mode accepts independent prompts; system prompts and previous
conversation turns are rejected rather than silently omitted.

`name` is the full API resource name, for example
`projects/your-project-id/locations/global/batchPredictionJobs/123456789`.
It includes the batch job ID and can be passed directly to the SDK to check
status later. With the same Google Cloud environment and credentials, this
separate command reads the saved receipt and makes a single status request:

```bash
python - <<'PY'
import json
from google import genai

with open("batch-job.json") as f:
    receipt = json.load(f)
with genai.Client() as client:
    job = client.batches.get(name=receipt["name"])
    print(job.model_dump_json(indent=2, exclude_none=True))
PY
```

This manual check is not run by the plugin and does not submit another job.

YouTube URLs are passed through to `fileData.fileUri`. **YouTube and agentic video
processing in batch have not been verified with a live job**; job acceptance does
not guarantee that individual video requests succeed. The official
[Cloud Storage batch guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference/new-job-from-cloud-storage)
documents JSONL input and GCS video examples. This plugin intentionally permits
YouTube URLs for subsequent live testing.

The caller needs permission to upload input objects and create batch jobs. The
project's AI Platform Service Agent needs to read the input JSONL and write to
the output bucket; configure access to both buckets if they differ. Enable the
required Google Cloud APIs and use locations supported by the selected model.
The plugin does not provision buckets or change IAM. Upload/submission failures
raise `llm.ModelError` with the artifact URIs. Uploaded JSONL is retained even
if submission fails; check Cloud Console before retrying an ambiguous submission
failure to avoid duplicate paid jobs.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-gemini-youtube
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
python -m pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```

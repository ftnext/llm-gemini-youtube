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

Supported models are `gemini-3.7-flash-yt`, `gemini-3.6-flash-yt`, and `gemini-3.5-flash-lite-yt`. Configure the API key using LLM's standard key mechanism:

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

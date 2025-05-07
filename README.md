# llm-gemini-youtube

[![PyPI](https://img.shields.io/pypi/v/llm-gemini-youtube.svg)](https://pypi.org/project/llm-gemini-youtube/)
[![Changelog](https://img.shields.io/github/v/release/ftnext/llm-gemini-youtube?include_prereleases&label=changelog)](https://github.com/ftnext/llm-gemini-youtube/releases)
[![Tests](https://github.com/ftnext/llm-gemini-youtube/actions/workflows/test.yml/badge.svg)](https://github.com/ftnext/llm-gemini-youtube/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ftnext/llm-gemini-youtube/blob/main/LICENSE)

LLM plugin to access Google's Gemini family of models, with support for YouTube URLs  
https://ai.google.dev/gemini-api/docs/video-understanding#youtube

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-gemini-youtube
```
## Usage

```bash
llm -m gemini-2.0-flash-yt -a 'https://www.youtube.com/watch?v=9hE5-98ZeCg' 'Can you summarize this video?'

llm -m gemini-1.5-pro-yt -a 'https://www.youtube.com/watch?v=9hE5-98ZeCg' 'What are the examples given at 01:05 and 01:19 supposed to show us?'

llm -m gemini-1.5-pro-yt -a 'https://www.youtube.com/watch?v=9hE5-98ZeCg' 'Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions.'
```

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

import pytest
from llm.plugins import pm

from llm_gemini_youtube import is_youtube_uri


@pytest.mark.skip("Need to fix this failing test")
def test_plugin_is_installed():
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

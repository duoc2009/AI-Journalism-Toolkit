"""Tests for the news scraper module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.scraper.news_scraper import Article, NewsScraper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_HTML_GENERIC = """
<html>
<head>
  <meta property="og:title" content="Test Article Title" />
  <meta property="article:published_time" content="2024-01-15T10:00:00Z" />
  <meta name="author" content="Jane Doe" />
</head>
<body>
  <article>
    <h1>Test Article Title</h1>
    <p>This is the first paragraph of the test article body text, which is long enough.</p>
    <p>This is the second paragraph of the test article body text, which is long enough.</p>
    <p>Short</p>
  </article>
</body>
</html>
"""

FAKE_HTML_BBC = """
<html>
<head>
  <meta property="og:title" content="BBC Article Title" />
  <meta property="article:published_time" content="2024-02-01T08:30:00Z" />
  <meta property="article:author" content="John Smith" />
</head>
<body>
  <article>
    <p>BBC paragraph one with enough text to pass the filter.</p>
    <p>BBC paragraph two with enough text to pass the filter.</p>
  </article>
</body>
</html>
"""


def _make_mock_response(html: str, status_code: int = 200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = html
    mock.raise_for_status = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# Article dataclass tests
# ---------------------------------------------------------------------------

class TestArticle:
    def test_default_source_from_url(self):
        art = Article(url="https://example.com/article")
        assert art.source == "example.com"

    def test_explicit_source_not_overridden(self):
        art = Article(url="https://example.com/article", source="My Source")
        assert art.source == "My Source"

    def test_default_fields(self):
        art = Article(url="https://example.com/article")
        assert art.title == ""
        assert art.body == ""
        assert art.tags == []


# ---------------------------------------------------------------------------
# NewsScraper tests
# ---------------------------------------------------------------------------

class TestNewsScraper:
    def setup_method(self):
        self.scraper = NewsScraper()

    def _mock_get(self, html, status_code=200):
        resp = _make_mock_response(html, status_code)
        self.scraper._session.get = MagicMock(return_value=resp)
        return resp

    # --- generic parser ---

    def test_scrape_generic_title(self):
        self._mock_get(FAKE_HTML_GENERIC)
        art = self.scraper.scrape("https://example.com/news/1")
        assert art.title == "Test Article Title"

    def test_scrape_generic_body(self):
        self._mock_get(FAKE_HTML_GENERIC)
        art = self.scraper.scrape("https://example.com/news/1")
        assert "first paragraph" in art.body
        assert "second paragraph" in art.body
        # Short paragraph (<40 chars) should be filtered out
        assert "Short" not in art.body

    def test_scrape_generic_author(self):
        self._mock_get(FAKE_HTML_GENERIC)
        art = self.scraper.scrape("https://example.com/news/1")
        assert art.author == "Jane Doe"

    def test_scrape_generic_date(self):
        self._mock_get(FAKE_HTML_GENERIC)
        art = self.scraper.scrape("https://example.com/news/1")
        assert art.published_date == "2024-01-15T10:00:00Z"

    # --- BBC-specific parser ---

    def test_scrape_bbc(self):
        self._mock_get(FAKE_HTML_BBC)
        art = self.scraper.scrape("https://www.bbc.com/news/uk-12345")
        assert art.title == "BBC Article Title"
        assert "BBC paragraph one" in art.body

    # --- scrape_many ---

    def test_scrape_many_success(self):
        self._mock_get(FAKE_HTML_GENERIC)
        articles = self.scraper.scrape_many(
            ["https://example.com/1", "https://example.com/2"]
        )
        assert len(articles) == 2

    def test_scrape_many_skips_failed_urls(self):
        good_resp = _make_mock_response(FAKE_HTML_GENERIC)
        bad_resp = _make_mock_response("", 404)
        bad_resp.raise_for_status = MagicMock(side_effect=Exception("404"))

        self.scraper._session.get = MagicMock(side_effect=[good_resp, bad_resp])
        articles = self.scraper.scrape_many(
            ["https://example.com/good", "https://example.com/bad"]
        )
        assert len(articles) == 1

    # --- HTTP error propagation ---

    def test_scrape_raises_on_http_error(self):
        resp = _make_mock_response("", 500)
        resp.raise_for_status = MagicMock(side_effect=Exception("500 Server Error"))
        self.scraper._session.get = MagicMock(return_value=resp)
        with pytest.raises(Exception, match="500"):
            self.scraper.scrape("https://example.com/error")

"""Tests for the infographic generator module."""

from __future__ import annotations

import pytest

from src.visualizer.infographic_generator import (
    InfographicGenerator,
    _sentiment_score,
    _tokenize,
)

SAMPLE_TEXT = (
    "Artificial intelligence transforms journalism and newsrooms worldwide. "
    "Machine learning helps journalists detect patterns in large datasets. "
    "Reporters are collaborating with AI-powered tools to accelerate investigations."
)

SAMPLE_ARTICLES = [
    {"title": "Great win for the economy", "body": SAMPLE_TEXT, "source": "BBC", "published_date": "2024-01-01"},
    {"title": "Market crash threatens growth", "body": "Stocks fell sharply today.", "source": "Reuters", "published_date": "2024-01-02"},
    {"title": "Neutral headline here", "body": "A report was released.", "source": "BBC", "published_date": "2024-01-03"},
]


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_removes_stop_words(self):
        tokens = _tokenize("the cat sat on the mat")
        assert "the" not in tokens
        assert "on" not in tokens
        assert "cat" in tokens

    def test_lowercases(self):
        tokens = _tokenize("Artificial Intelligence")
        assert "artificial" in tokens
        assert "intelligence" in tokens

    def test_filters_short_words(self):
        # Words shorter than 3 chars should be excluded by the regex \b[a-z]{3,}\b
        tokens = _tokenize("AI is great")
        assert "ai" not in tokens  # only 2 chars


class TestSentimentScore:
    def test_positive_text(self):
        score = _sentiment_score("Great win for the economy and strong growth")
        assert score > 0

    def test_negative_text(self):
        score = _sentiment_score("Market crash and economic failure disaster")
        assert score < 0

    def test_neutral_text(self):
        score = _sentiment_score("A report was released today about the weather")
        assert score == 0.0

    def test_empty_string(self):
        assert _sentiment_score("") == 0.0


# ---------------------------------------------------------------------------
# InfographicGenerator tests
# ---------------------------------------------------------------------------

class TestInfographicGenerator:
    def setup_method(self):
        self.viz = InfographicGenerator()

    # --- word_frequency_chart ---

    def test_word_frequency_returns_bytes(self):
        result = self.viz.word_frequency_chart(SAMPLE_TEXT)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"  # PNG magic bytes

    def test_word_frequency_raises_on_empty_text(self):
        with pytest.raises(ValueError, match="No words found"):
            self.viz.word_frequency_chart("the a an of")

    def test_word_frequency_saves_file(self, tmp_path):
        path = str(tmp_path / "freq.png")
        self.viz.word_frequency_chart(SAMPLE_TEXT, save_path=path)
        import os
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    # --- keyword_trend_chart ---

    def test_keyword_trend_returns_bytes(self):
        result = self.viz.keyword_trend_chart(
            SAMPLE_ARTICLES,
            keywords=["economy", "intelligence"],
        )
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    # --- source_distribution_chart ---

    def test_source_distribution_returns_bytes(self):
        result = self.viz.source_distribution_chart(SAMPLE_ARTICLES)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_source_distribution_single_source(self):
        arts = [{"title": "A", "body": "b", "source": "BBC", "published_date": ""}]
        result = self.viz.source_distribution_chart(arts)
        assert isinstance(result, bytes)

    # --- sentiment_chart ---

    def test_sentiment_chart_returns_bytes(self):
        result = self.viz.sentiment_chart(SAMPLE_ARTICLES)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_sentiment_chart_with_article_objects(self):
        from src.scraper.news_scraper import Article

        articles = [
            Article(url="https://example.com/1", title="Great economic win"),
            Article(url="https://example.com/2", title="Market crash and loss"),
        ]
        result = self.viz.sentiment_chart(articles)
        assert isinstance(result, bytes)

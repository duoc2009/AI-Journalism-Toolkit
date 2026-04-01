"""Tests for the text summarizer module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.summarizer.text_summarizer import (
    TextSummarizer,
    _compute_tf_idf,
    _split_sentences,
)

SAMPLE_TEXT = (
    "Artificial intelligence is transforming journalism. "
    "Newsrooms are using machine learning to automate fact-checking. "
    "Data-driven stories now rely on algorithms to find patterns in large datasets. "
    "Reporters collaborate with AI tools to speed up investigations. "
    "The future of journalism will involve human-AI partnerships."
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_basic_split(self):
        text = "Hello world. This is a test. Another sentence."
        result = _split_sentences(text)
        assert len(result) == 3

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_single_sentence(self):
        result = _split_sentences("Only one sentence here.")
        assert len(result) == 1

    def test_exclamation_and_question(self):
        text = "Is this working? Yes it is! Great."
        result = _split_sentences(text)
        assert len(result) == 3


class TestComputeTfIdf:
    def test_returns_score_per_sentence(self):
        sentences = ["hello world", "world foo bar", "baz qux"]
        scores = _compute_tf_idf(sentences)
        assert len(scores) == len(sentences)

    def test_scores_are_non_negative(self):
        sentences = ["artificial intelligence transforms journalism", "data and algorithms"]
        scores = _compute_tf_idf(sentences)
        assert all(s >= 0 for s in scores)


# ---------------------------------------------------------------------------
# TextSummarizer tests
# ---------------------------------------------------------------------------

class TestTextSummarizer:
    # --- extractive (offline) mode ---

    def test_extractive_returns_string(self):
        summarizer = TextSummarizer(api_key=None)
        result = summarizer.summarize(SAMPLE_TEXT)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extractive_fewer_sentences_than_max(self):
        """When input has fewer sentences than max, all are returned."""
        summarizer = TextSummarizer(api_key=None)
        short_text = "One sentence. Two sentences."
        result = summarizer.summarize(short_text, max_sentences=5)
        assert "One sentence" in result
        assert "Two sentences" in result

    def test_extractive_empty_input(self):
        summarizer = TextSummarizer(api_key=None)
        assert summarizer.summarize("") == ""
        assert summarizer.summarize("   ") == ""

    def test_summarize_bulk(self):
        summarizer = TextSummarizer(api_key=None)
        texts = [SAMPLE_TEXT, "Short text. Another sentence."]
        results = summarizer.summarize_bulk(texts)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    # --- LLM mode (mocked) ---

    def test_llm_summarize_called(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "LLM summary of the article."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        summarizer = TextSummarizer.__new__(TextSummarizer)
        summarizer.model = "gpt-4o-mini"
        summarizer.max_tokens = 256
        summarizer.temperature = 0.3
        summarizer._client = mock_client

        result = summarizer.summarize(SAMPLE_TEXT)
        assert result == "LLM summary of the article."
        mock_client.chat.completions.create.assert_called_once()

    def test_llm_fallback_on_exception(self):
        """When the LLM call fails, extractive summary is returned."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        summarizer = TextSummarizer.__new__(TextSummarizer)
        summarizer.model = "gpt-4o-mini"
        summarizer.max_tokens = 256
        summarizer.temperature = 0.3
        summarizer._client = mock_client

        result = summarizer.summarize(SAMPLE_TEXT)
        # Should get an extractive result, not raise
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_api_key_uses_extractive(self):
        """With no key in env and api_key=None, client is None."""
        with patch.dict("os.environ", {}, clear=True):
            # Ensure OPENAI_API_KEY is absent
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            summarizer = TextSummarizer(api_key=None)
        assert summarizer._client is None

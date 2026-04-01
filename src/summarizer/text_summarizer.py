"""Text summarization module.

Provides two strategies:

1. **LLM-based** (default when ``OPENAI_API_KEY`` is present in the
   environment): sends the article text to the OpenAI Chat Completions
   API and returns the model-generated summary.

2. **Extractive** (fallback / offline mode): selects the highest-scoring
   sentences from the article using a simple TF-IDF-inspired heuristic.
   Requires no external API key and no heavy ML dependencies.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_TEMPERATURE = 0.3
_EXTRACTIVE_SENTENCE_COUNT = 3


class TextSummarizer:
    """Summarise news article text using an LLM or an extractive fallback.

    Parameters
    ----------
    api_key:
        OpenAI API key.  If *None* the value of the ``OPENAI_API_KEY``
        environment variable is used.  When neither is available the
        summarizer silently falls back to the extractive strategy.
    model:
        OpenAI chat model to use (e.g. ``"gpt-4o-mini"``).
    max_tokens:
        Maximum number of tokens in the LLM response.
    temperature:
        Sampling temperature for the LLM.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = self._build_client(api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, text: str, max_sentences: int = _EXTRACTIVE_SENTENCE_COUNT) -> str:
        """Return a summary of *text*.

        If an OpenAI client is available the LLM strategy is used;
        otherwise the extractive strategy is applied.

        Parameters
        ----------
        text:
            Full article body to summarise.
        max_sentences:
            Number of sentences to include when the extractive fallback
            is used (ignored for LLM mode).

        Returns
        -------
        str
            The generated summary.
        """
        if not text or not text.strip():
            return ""
        if self._client is not None:
            return self._llm_summarize(text)
        return self._extractive_summarize(text, max_sentences)

    def summarize_bulk(
        self, texts: List[str], max_sentences: int = _EXTRACTIVE_SENTENCE_COUNT
    ) -> List[str]:
        """Summarise a list of article bodies.

        Parameters
        ----------
        texts:
            List of article body strings.
        max_sentences:
            Sentence count for the extractive fallback.

        Returns
        -------
        list[str]
            Summary for each input text, in the same order.
        """
        return [self.summarize(t, max_sentences) for t in texts]

    # ------------------------------------------------------------------
    # LLM strategy
    # ------------------------------------------------------------------

    def _build_client(self, api_key: Optional[str]):
        """Return an ``openai.OpenAI`` client or *None* if unavailable."""
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            logger.info("No OpenAI API key found – using extractive summarizer.")
            return None
        try:
            import openai  # noqa: PLC0415

            return openai.OpenAI(api_key=key)
        except ImportError:
            logger.warning("openai package not installed – using extractive summarizer.")
            return None

    def _llm_summarize(self, text: str) -> str:
        """Call the OpenAI Chat Completions API."""
        system_prompt = (
            "You are an expert journalist editor. "
            "Summarise the following news article in 3-5 concise sentences, "
            "preserving the key facts, named entities, and main narrative. "
            "Do not add information that is not in the article."
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM summarization failed (%s); falling back to extractive.", exc)
            return self._extractive_summarize(text)

    # ------------------------------------------------------------------
    # Extractive strategy
    # ------------------------------------------------------------------

    @staticmethod
    def _extractive_summarize(text: str, max_sentences: int = _EXTRACTIVE_SENTENCE_COUNT) -> str:
        """Select the top *max_sentences* sentences by TF-IDF score."""
        sentences = _split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        tf_idf = _compute_tf_idf(sentences)
        scored = sorted(enumerate(sentences), key=lambda x: tf_idf[x[0]], reverse=True)
        top_indices = sorted(idx for idx, _ in scored[:max_sentences])
        return " ".join(sentences[i] for i in top_indices)


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Naïve sentence splitter that handles common abbreviations."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _word_freq(sentence: str) -> dict:
    words = re.findall(r"\b[a-z]{2,}\b", sentence.lower())
    freq: dict = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq


def _compute_tf_idf(sentences: List[str]) -> List[float]:
    """Return a TF-IDF-inspired score for each sentence."""
    n = len(sentences)
    # Document frequency per word (across sentences)
    df: dict = {}
    freqs = [_word_freq(s) for s in sentences]
    for freq in freqs:
        for w in freq:
            df[w] = df.get(w, 0) + 1

    scores: List[float] = []
    for freq in freqs:
        total_words = sum(freq.values()) or 1
        score = 0.0
        for w, cnt in freq.items():
            tf = cnt / total_words
            idf = math.log((n + 1) / (df[w] + 1)) + 1
            score += tf * idf
        scores.append(score)
    return scores

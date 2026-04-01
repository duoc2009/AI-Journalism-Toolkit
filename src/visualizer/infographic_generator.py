"""Infographic / data-visualisation generator for news articles.

Produces publication-ready charts that can be dropped straight into a
journalistic piece or digital infographic:

* **Word-frequency bar chart** – most common words in an article body.
* **Keyword trend line chart** – how often a set of keywords appears
  across a collection of articles (e.g. over time).
* **Source distribution pie chart** – share of articles by news source.
* **Sentiment bar chart** – polarity score per article title (using a
  lightweight lexicon so no external model is needed).

All charts are rendered via *matplotlib* and can be saved to disk or
returned as in-memory ``bytes`` (PNG).
"""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend – safe for scripts and tests

logger = logging.getLogger(__name__)

# Minimal English stop-word list (keeps the module dependency-free)
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "not", "no", "it", "its",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "they",
    "them", "their", "our", "your", "my", "his", "her", "as", "if", "so",
    "up", "out", "about", "into", "over", "after", "said", "also",
}

# Tiny sentiment lexicon (positive / negative polarity words)
_POS_WORDS = {
    "good", "great", "win", "gains", "surge", "rise", "boost", "strong",
    "success", "positive", "improve", "growth", "record", "best", "high",
    "advance", "support", "agree", "deal", "peace",
}
_NEG_WORDS = {
    "bad", "fail", "loss", "crash", "fall", "drop", "weak", "crisis",
    "negative", "decline", "worst", "low", "attack", "war", "conflict",
    "disaster", "death", "dead", "kill", "threat", "risk", "fear",
}


class InfographicGenerator:
    """Generate charts and infographics from scraped news data.

    Parameters
    ----------
    figsize:
        Default figure size ``(width, height)`` in inches.
    dpi:
        Resolution for saved images.
    style:
        Matplotlib style sheet name (e.g. ``"ggplot"``, ``"seaborn-v0_8"``).
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150,
        style: str = "ggplot",
    ) -> None:
        self.figsize = figsize
        self.dpi = dpi
        try:
            plt.style.use(style)
        except OSError:
            logger.warning("Matplotlib style '%s' not found; using default.", style)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def word_frequency_chart(
        self,
        text: str,
        top_n: int = 20,
        title: str = "Top Words in Article",
        save_path: Optional[str] = None,
    ) -> bytes:
        """Bar chart of the most frequent words in *text*.

        Parameters
        ----------
        text:
            Article body to analyse.
        top_n:
            Number of top words to display.
        title:
            Chart title.
        save_path:
            If provided, the chart is also written to this file path.

        Returns
        -------
        bytes
            PNG image data.
        """
        words = _tokenize(text)
        counter = Counter(words)
        if not counter:
            raise ValueError("No words found in the provided text.")
        labels, values = zip(*counter.most_common(top_n))

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.barh(list(reversed(labels)), list(reversed(values)), color="steelblue")
        ax.bar_label(bars, padding=3, fontsize=8)
        ax.set_xlabel("Frequency")
        ax.set_title(title)
        plt.tight_layout()

        return self._save_or_return(fig, save_path)

    def keyword_trend_chart(
        self,
        articles: List[Dict],
        keywords: Sequence[str],
        date_field: str = "published_date",
        title: str = "Keyword Trend Over Time",
        save_path: Optional[str] = None,
    ) -> bytes:
        """Line chart showing how often each keyword appears across articles.

        Parameters
        ----------
        articles:
            List of dicts (or :class:`~src.scraper.Article`-like objects
            converted to dicts) with at least ``body`` and *date_field* keys.
        keywords:
            Keywords to track.
        date_field:
            Key used to read the publication date from each article dict.
        title:
            Chart title.
        save_path:
            Optional file path to save the chart.

        Returns
        -------
        bytes
            PNG image data.
        """
        records = []
        for art in articles:
            body = art.get("body", "") if isinstance(art, dict) else getattr(art, "body", "")
            date = art.get(date_field, "") if isinstance(art, dict) else getattr(art, date_field, "")
            words = set(_tokenize(body))
            row = {"date": date or "unknown"}
            for kw in keywords:
                row[kw] = 1 if kw.lower() in words else 0
            records.append(row)

        df = pd.DataFrame(records)
        df = df.groupby("date")[list(keywords)].sum().sort_index()

        fig, ax = plt.subplots(figsize=self.figsize)
        for kw in keywords:
            if kw in df.columns:
                ax.plot(df.index, df[kw], marker="o", label=kw)
        ax.set_xlabel("Date")
        ax.set_ylabel("Article count containing keyword")
        ax.set_title(title)
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        return self._save_or_return(fig, save_path)

    def source_distribution_chart(
        self,
        articles: List[Dict],
        title: str = "Articles by Source",
        save_path: Optional[str] = None,
    ) -> bytes:
        """Pie chart of article counts per news source.

        Parameters
        ----------
        articles:
            List of dicts or Article objects with a ``source`` field.
        title:
            Chart title.
        save_path:
            Optional file path to save the chart.

        Returns
        -------
        bytes
            PNG image data.
        """
        sources = [
            (art.get("source") if isinstance(art, dict) else getattr(art, "source", "unknown"))
            or "unknown"
            for art in articles
        ]
        counter = Counter(sources)
        labels = list(counter.keys())
        sizes = list(counter.values())

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.85,
        )
        ax.set_title(title)
        plt.tight_layout()

        return self._save_or_return(fig, save_path)

    def sentiment_chart(
        self,
        articles: List[Dict],
        title: str = "Article Sentiment by Title",
        save_path: Optional[str] = None,
    ) -> bytes:
        """Horizontal bar chart of per-article sentiment polarity.

        Sentiment is computed with a lightweight lexicon: each word in
        the title is checked against positive / negative word lists and
        a score in ``[-1, 1]`` is returned.

        Parameters
        ----------
        articles:
            List of dicts or Article objects with a ``title`` field.
        title:
            Chart title.
        save_path:
            Optional file path to save the chart.

        Returns
        -------
        bytes
            PNG image data.
        """
        labels: List[str] = []
        scores: List[float] = []

        for art in articles:
            art_title = (
                art.get("title") if isinstance(art, dict) else getattr(art, "title", "")
            ) or ""
            score = _sentiment_score(art_title)
            # Truncate long titles for readability
            short_title = art_title[:60] + "…" if len(art_title) > 60 else art_title
            labels.append(short_title or "(no title)")
            scores.append(score)

        colors = ["#2ecc71" if s >= 0 else "#e74c3c" for s in scores]

        fig, ax = plt.subplots(figsize=self.figsize)
        y_pos = range(len(labels))
        ax.barh(list(y_pos), scores, color=colors)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Sentiment score")
        ax.set_title(title)
        plt.tight_layout()

        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_or_return(self, fig: plt.Figure, save_path: Optional[str]) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi)
        plt.close(fig)
        buf.seek(0)
        data = buf.read()
        if save_path:
            with open(save_path, "wb") as fh:
                fh.write(data)
            logger.info("Chart saved to %s", save_path)
        return data


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lower-case, alphabetic tokens with stop-words removed."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in _STOP_WORDS]


def _sentiment_score(text: str) -> float:
    """Return a polarity score in [-1, 1] using the built-in lexicon.

    The score is ``(positive_count - negative_count) / total_word_count``.
    Because ``positive_count + negative_count <= total_word_count``, the
    result is always in the closed interval ``[-1, 1]``.
    """
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)
    return (pos - neg) / len(words)

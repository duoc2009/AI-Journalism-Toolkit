"""Web scraper for news articles.

Uses *requests* for HTTP and *BeautifulSoup* for HTML parsing.
Supports generic article extraction as well as a small set of
site-specific extraction strategies (BBC, Reuters, AP News).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AIJournalismToolkit/1.0; "
        "+https://github.com/duoc2009/AI-Journalism-Toolkit)"
    )
}
_REQUEST_TIMEOUT = 15  # seconds


@dataclass
class Article:
    """Represents a single scraped news article."""

    url: str
    title: str = ""
    body: str = ""
    author: str = ""
    published_date: str = ""
    source: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source = self.source or urlparse(self.url).netloc


class NewsScraper:
    """Scrape one or more news article URLs and return :class:`Article` objects.

    Parameters
    ----------
    headers:
        Optional HTTP headers to send with every request.  Defaults to a
        generic browser-like User-Agent so that most news sites respond
        with full HTML.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        headers: Optional[dict] = None,
        timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        self.headers = headers or _DEFAULT_HEADERS
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(self.headers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self, url: str) -> Article:
        """Scrape a single article URL.

        Parameters
        ----------
        url:
            Full URL of the news article to scrape.

        Returns
        -------
        Article
            Populated :class:`Article` dataclass instance.

        Raises
        ------
        requests.HTTPError
            When the server returns a non-2xx status code.
        """
        logger.info("Scraping article: %s", url)
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        return self._parse(url, soup)

    def scrape_many(self, urls: List[str]) -> List[Article]:
        """Scrape multiple article URLs.

        Failed URLs are logged as warnings and skipped rather than
        raising an exception, so a single bad link does not abort the
        whole batch.

        Parameters
        ----------
        urls:
            List of article URLs to scrape.

        Returns
        -------
        list[Article]
            Successfully scraped articles (may be shorter than *urls*).
        """
        articles: List[Article] = []
        for url in urls:
            try:
                articles.append(self.scrape(url))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to scrape %s: %s", url, exc)
        return articles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, url: str, soup: BeautifulSoup) -> Article:
        """Route to a site-specific or generic parser."""
        hostname = urlparse(url).hostname or ""
        if "bbc" in hostname:
            return self._parse_bbc(url, soup)
        if "reuters" in hostname:
            return self._parse_reuters(url, soup)
        if "apnews" in hostname:
            return self._parse_ap(url, soup)
        return self._parse_generic(url, soup)

    # --- site-specific parsers ----------------------------------------

    def _parse_bbc(self, url: str, soup: BeautifulSoup) -> Article:
        title = self._meta_or_tag(soup, "og:title", "h1")
        body = self._join_paragraphs(soup.select("article p"))
        author = self._meta_text(soup, "article:author")
        date = self._meta_text(soup, "article:published_time")
        return Article(url=url, title=title, body=body, author=author, published_date=date)

    def _parse_reuters(self, url: str, soup: BeautifulSoup) -> Article:
        title = self._meta_or_tag(soup, "og:title", "h1")
        body = self._join_paragraphs(soup.select("p[class*='paragraph']"))
        date = self._meta_text(soup, "article:published_time")
        return Article(url=url, title=title, body=body, published_date=date)

    def _parse_ap(self, url: str, soup: BeautifulSoup) -> Article:
        title = self._meta_or_tag(soup, "og:title", "h1")
        body = self._join_paragraphs(soup.select("div.Article p"))
        date = self._meta_text(soup, "article:published_time")
        return Article(url=url, title=title, body=body, published_date=date)

    def _parse_generic(self, url: str, soup: BeautifulSoup) -> Article:
        """Best-effort extraction for any news site."""
        title = self._meta_or_tag(soup, "og:title", "h1")
        author = (
            self._meta_text(soup, "author")
            or self._meta_text(soup, "article:author")
            or self._tag_text(soup, "[rel='author']")
            or self._tag_text(soup, ".author")
        )
        date = (
            self._meta_text(soup, "article:published_time")
            or self._meta_text(soup, "date")
            or self._tag_text(soup, "time")
        )
        tags = [
            tag.get_text(strip=True)
            for tag in soup.select("a[rel='tag'], .tag, .label")
        ]

        # Prefer <article> element; fall back to largest <div>
        article_el = soup.find("article")
        if article_el:
            paragraphs = article_el.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        # Filter out very short strings (nav / footer debris)
        paragraphs = [p for p in paragraphs if len(p.get_text(strip=True)) > 40]
        body = self._join_paragraphs(paragraphs)

        return Article(
            url=url,
            title=title,
            body=body,
            author=author,
            published_date=date,
            tags=tags,
        )

    # --- utility helpers ----------------------------------------------

    @staticmethod
    def _meta_text(soup: BeautifulSoup, prop: str) -> str:
        tag = soup.find("meta", attrs={"property": prop}) or soup.find(
            "meta", attrs={"name": prop}
        )
        if tag and tag.get("content"):
            return tag["content"].strip()
        return ""

    @staticmethod
    def _tag_text(soup: BeautifulSoup, selector: str) -> str:
        tag = soup.select_one(selector)
        return tag.get_text(strip=True) if tag else ""

    @staticmethod
    def _meta_or_tag(soup: BeautifulSoup, meta_prop: str, tag_name: str) -> str:
        meta = soup.find("meta", attrs={"property": meta_prop}) or soup.find(
            "meta", attrs={"name": meta_prop}
        )
        if meta and meta.get("content"):
            return meta["content"].strip()
        tag = soup.find(tag_name)
        return tag.get_text(strip=True) if tag else ""

    @staticmethod
    def _join_paragraphs(tags) -> str:
        return "\n\n".join(t.get_text(separator=" ", strip=True) for t in tags)

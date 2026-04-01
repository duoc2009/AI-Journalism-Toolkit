"""AI Journalism Toolkit – command-line entry point.

Usage examples
--------------
# Scrape, summarise, and visualise a single article
python main.py --urls https://example.com/article1

# Pass multiple URLs
python main.py --urls https://example.com/a1 https://example.com/a2

# Save generated charts to a directory
python main.py --urls https://example.com/article1 --output-dir ./charts

# Use a specific OpenAI model
python main.py --urls https://example.com/article1 --model gpt-4o

# Run in offline / extractive-only mode (no API key required)
python main.py --urls https://example.com/article1 --offline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from src.scraper import NewsScraper
from src.summarizer import TextSummarizer
from src.visualizer import InfographicGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Journalism Toolkit: scrape, summarise, and visualise news articles."
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        required=True,
        metavar="URL",
        help="One or more news article URLs to process.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory where chart images are saved (default: current directory).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use for summarisation (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable LLM summarisation and use the extractive fallback.",
    )
    parser.add_argument(
        "--top-words",
        type=int,
        default=20,
        metavar="N",
        help="Number of top words to show in the frequency chart (default: 20).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scrape articles
    logger.info("Scraping %d URL(s)…", len(args.urls))
    scraper = NewsScraper()
    articles = scraper.scrape_many(args.urls)
    if not articles:
        logger.error("No articles could be scraped. Exiting.")
        return 1
    logger.info("Scraped %d article(s).", len(articles))

    # 2. Summarise
    api_key = None if args.offline else os.environ.get("OPENAI_API_KEY")
    summarizer = TextSummarizer(api_key=api_key, model=args.model)
    summaries = summarizer.summarize_bulk([a.body for a in articles])

    results = []
    for article, summary in zip(articles, summaries):
        results.append(
            {
                "url": article.url,
                "title": article.title,
                "source": article.source,
                "published_date": article.published_date,
                "author": article.author,
                "summary": summary,
            }
        )
        print("\n" + "=" * 70)
        print(f"Title  : {article.title}")
        print(f"Source : {article.source}")
        print(f"Date   : {article.published_date}")
        print(f"Author : {article.author}")
        print(f"Summary:\n{summary}")

    # 3. Generate visualisations
    viz = InfographicGenerator()
    article_dicts = [
        {"title": a.title, "body": a.body, "source": a.source, "published_date": a.published_date}
        for a in articles
    ]

    combined_body = "\n\n".join(a.body for a in articles if a.body)
    if combined_body:
        path = str(output_dir / "word_frequency.png")
        viz.word_frequency_chart(combined_body, top_n=args.top_words, save_path=path)
        logger.info("Word-frequency chart saved to %s", path)

    if len(articles) >= 2:
        path = str(output_dir / "source_distribution.png")
        viz.source_distribution_chart(article_dicts, save_path=path)
        logger.info("Source-distribution chart saved to %s", path)

    path = str(output_dir / "sentiment.png")
    viz.sentiment_chart(article_dicts, save_path=path)
    logger.info("Sentiment chart saved to %s", path)

    # Write JSON results
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Results written to %s", json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

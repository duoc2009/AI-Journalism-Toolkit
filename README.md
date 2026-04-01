# AI Journalism Toolkit

An open-source Python toolkit for journalists to automate news gathering,
text summarisation, and data-driven storytelling with AI.

---

## Features

| Module | Description |
|--------|-------------|
| `src/scraper` | Web scraper for news articles (requests + BeautifulSoup) |
| `src/summarizer` | LLM-based text summarisation (OpenAI) with extractive fallback |
| `src/visualizer` | Data-visualisation / infographic generator (matplotlib) |

---

## Project Structure

```
AI-Journalism-Toolkit/
├── main.py                          # CLI entry point
├── requirements.txt
├── src/
│   ├── scraper/
│   │   └── news_scraper.py          # NewsScraper, Article
│   ├── summarizer/
│   │   └── text_summarizer.py       # TextSummarizer (LLM + extractive)
│   └── visualizer/
│       └── infographic_generator.py # InfographicGenerator
└── tests/
    ├── test_scraper.py
    ├── test_summarizer.py
    └── test_visualizer.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Command-line

```bash
# Scrape, summarise, and visualise one or more articles
python main.py --urls https://example.com/article1 https://example.com/article2

# Save charts to a specific directory
python main.py --urls https://example.com/article1 --output-dir ./charts

# Use a specific OpenAI model
python main.py --urls https://example.com/article1 --model gpt-4o

# Offline mode – extractive summarisation, no API key required
python main.py --urls https://example.com/article1 --offline
```

Set `OPENAI_API_KEY` in your environment to enable LLM summarisation:

```bash
export OPENAI_API_KEY="sk-..."
python main.py --urls https://example.com/article1
```

### Python API

#### Web Scraper

```python
from src.scraper import NewsScraper

scraper = NewsScraper()
article = scraper.scrape("https://example.com/article")
print(article.title)
print(article.body)

# Scrape multiple URLs (failed URLs are skipped with a warning)
articles = scraper.scrape_many([url1, url2, url3])
```

#### Text Summarizer

```python
from src.summarizer import TextSummarizer

# LLM mode (requires OPENAI_API_KEY)
summarizer = TextSummarizer()
summary = summarizer.summarize(article.body)

# Extractive mode (no API key needed)
summarizer = TextSummarizer(api_key=None)
summary = summarizer.summarize(article.body, max_sentences=3)

# Batch summarisation
summaries = summarizer.summarize_bulk([a.body for a in articles])
```

#### Infographic Generator

```python
from src.visualizer import InfographicGenerator

viz = InfographicGenerator()

# Word-frequency bar chart
png_bytes = viz.word_frequency_chart(article.body, save_path="freq.png")

# Keyword trend over time
png_bytes = viz.keyword_trend_chart(article_dicts, keywords=["AI", "economy"])

# Source distribution pie chart
png_bytes = viz.source_distribution_chart(article_dicts, save_path="sources.png")

# Sentiment bar chart
png_bytes = viz.sentiment_chart(article_dicts, save_path="sentiment.png")
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

See [LICENSE](LICENSE).

# Automated OpenInsider Evaluator

A Streamlit application for scraping insider-trading data from OpenInsider,
enriching it with yfinance market data, and ranking results with customizable weighting.

## Features
- Scrape multi-page tables from OpenInsider using requests/BeautifulSoup or Selenium.
- Optionally roll up entries to per-ticker features.
- Enrich scraped data with market capitalization and current price via yfinance.
- Score factors with adjustable weights and rank stocks.
- Save and load weight presets.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app_streamlit.py
```

## Notes
- The scraper targets OpenInsider tables; HTML changes may break parsing. Prefer CSV downloads when possible.
- Be polite and throttle requests with the delay setting.
- All factor inputs are percentile-normalized before weighting; MarketCap uses inverse percentiles (smaller = better).
- Title multipliers default: CEO 1.00, CFO 0.95, COO/President 0.90, Director 0.75, 10% Owner 0.60, Other Exec/Officer 0.50, Unknown 0.30.
- Optional timing bonus for recent trades.

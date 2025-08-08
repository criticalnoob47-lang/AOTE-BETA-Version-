# openinsider_scraper.py
# Requests/BS4 scraper + headless Selenium fallback for OpenInsider.
# Improvements:
# - Reuse a pooled requests.Session with retries for faster multi-page scrapes
# - Robust header mapping
# - Roll-up utilities to aggregate per-insider rows into per-ticker features

import time
import re
from typing import Optional, List, Dict
import urllib.parse as urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd

# ---------------- HTTP ----------------
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Referer": "http://openinsider.com/",
}

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def fetch_html(url: str, sleep_sec: float = 0.7, session: requests.Session = None) -> str:
    sess = session or _session()
    r = sess.get(url, timeout=30)
    r.raise_for_status()
    if sleep_sec:
        time.sleep(float(sleep_sec))
    return r.text

# ---------------- URL paging ----------------
def build_page_url(base_url: str, page: int) -> str:
    """
    OpenInsider uses either 'page=N' or 'p=N'. Detect and set the correct param.
    """
    parsed = urlparse.urlparse(base_url)
    q = dict(urlparse.parse_qsl(parsed.query, keep_blank_values=True))
    if "p" in q and "page" not in q:
        q["p"] = str(page)
    else:
        q["page"] = str(page)
        q.pop("p", None)
    new_query = urlparse.urlencode(q)
    return urlparse.urlunparse(parsed._replace(query=new_query))

# ---------------- parsing helpers ----------------
def _clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return re.sub(r"\s+", " ", s).strip()

def _to_float(val: Optional[str]) -> Optional[float]:
    if not val:
        return None
    v = (
        str(val)
        .replace("$", "").replace(",", "").replace("+", "").replace("%", "").replace(">", "").strip()
    )
    if v in {"", "-", "New"}:
        return None
    try:
        return float(v)
    except Exception:
        return None

def _to_int(val: Optional[str]) -> Optional[int]:
    if not val:
        return None
    v = str(val).replace(",", "").replace("+", "").replace(">", "").strip()
    if v in {"", "-"}:
        return None
    try:
        return int(v)
    except Exception:
        return None

def _norm_header(h: str) -> str:
    h = (h or "").replace("\xa0", " ")
    h = h.replace("Δ", "Delta")  # &Delta;Own -> DeltaOwn
    h = h.lower().strip()
    return re.sub(r"[^a-z0-9]+", "", h)

# Canonical name mapping
CANON_MAP = {
    "x": "X",
    "filingdate": "FilingDate",
    "tradedate": "TradeDate",
    "ticker": "Ticker",
    "companyname": "Company",
    "industry": "Industry",
    "insidername": "Insider",
    "title": "Title",
    "ins": "NumInsiders",
    "tradetype": "TradeType",
    "price": "TradePrice",
    "qty": "Qty",
    "owned": "Owned",
    "deltaown": "OwnershipChangePct",
    "value": "ValueUSD",
    "1d": "Perf1d",
    "1w": "Perf1w",
    "1m": "Perf1m",
    "6m": "Perf6m",
}

# ---------------- core parser ----------------
def parse_tinytable_dynamic(html: str) -> pd.DataFrame:
    """
    Parse OpenInsider 'tinytable' into a dataframe by reading headers dynamically.
    Works across different list pages (screener, latest, filtered lists).
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.tinytable")
    if not table:
        return pd.DataFrame()

    # Map headers to canonical names
    headers = []
    for th in table.select("thead tr th"):
        headers.append(_norm_header(th.get_text(" ", strip=True)))

    idx2key: Dict[int, str] = {}
    for i, h in enumerate(headers):
        canon = CANON_MAP.get(h)
        if canon:
            idx2key[i] = canon

    tbody = table.find("tbody") or table
    rows = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        row: Dict[str, Optional[str]] = {}
        for idx, key in idx2key.items():
            if idx >= len(tds):
                continue
            raw = _clean(tds[idx].get_text(" ", strip=True))
            if key in (
                "FilingDate", "TradeDate", "Ticker", "Company", "Industry",
                "TradeType", "Insider", "Title",
            ):
                row[key] = raw
            elif key in ("TradePrice", "ValueUSD", "OwnershipChangePct", "Perf1d", "Perf1w", "Perf1m", "Perf6m"):
                val = _to_float(raw)
                if key == "OwnershipChangePct" and (val is not None) and val > 1000:
                    val = 1000.0
                row[key] = val
            elif key in ("Qty", "Owned", "NumInsiders"):
                row[key] = _to_int(raw)
            else:
                row[key] = raw

        # Ticker cleanup
        if row.get("Ticker"):
            row["Ticker"] = re.sub(r"[^A-Z\.\-]", "", row["Ticker"].upper())

        rows.append(row)

    return pd.DataFrame(rows)

# ---------------- public scraping API ----------------
def scrape_openinsider(base_url: str, pages: int = 1, sleep_sec: float = 0.7) -> pd.DataFrame:
    """
    Requests/BS4 path (fast, ad/overlay-agnostic) with pooled connections.
    """
    sess = _session()
    all_df: List[pd.DataFrame] = []
    for p in range(1, int(pages) + 1):
        url = build_page_url(base_url, p)
        html = fetch_html(url, sleep_sec=sleep_sec, session=sess)
        df = parse_tinytable_dynamic(html)
        if df.empty:
            if p == 1:
                continue
            break
        all_df.append(df)
    if not all_df:
        return pd.DataFrame(columns=list(set(CANON_MAP.values())))
    return pd.concat(all_df, ignore_index=True)

# ---------------- Selenium fallback ----------------
def scrape_openinsider_selenium(base_url: str, pages: int = 1, sleep_sec: float = 0.0) -> pd.DataFrame:
    """
    Headless Chrome fallback. Includes light content blocking prefs.
    Fixes the common 'multiple values for options' error by using Service(..., options=opts).
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--blink-settings=imagesEnabled=false")
    opts.add_experimental_option(
        "prefs",
        {
            "profile.default_content_setting_values.notifications": 2,
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.popups": 2,
            "profile.managed_default_content_settings.ads": 2,
        },
    )
    opts.add_argument(f'--user-agent={DEFAULT_HEADERS["User-Agent"]}')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)

    all_df: List[pd.DataFrame] = []
    try:
        for p in range(1, int(pages) + 1):
            url = build_page_url(base_url, p)
            driver.get(url)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.tinytable"))
            )
            html = driver.page_source
            df = parse_tinytable_dynamic(html)
            if df.empty:
                break
            all_df.append(df)
            if sleep_sec:
                time.sleep(float(sleep_sec))
    finally:
        driver.quit()

    if not all_df:
        return pd.DataFrame(columns=list(set(CANON_MAP.values())))
    return pd.concat(all_df, ignore_index=True)

# ---------------- Roll-up utilities ----------------
TITLE_MAP_DEFAULT = {
    "CEO": 1.00, "CFO": 0.95, "COO": 0.90, "PRESIDENT": 0.90, "CHAIR": 0.90,
    "PRES": 0.90, "DIR": 0.75, "DIRECTOR": 0.75, "10%": 0.60, "TENPCT": 0.60,
    "OWNER": 0.60, "VP": 0.50, "VICE": 0.50, "OFFICER": 0.50,
    "GENERAL COUNSEL": 0.50, "GC": 0.50, "OTHER": 0.30,
}

def _title_key(title: Optional[str]) -> str:
    t = (title or "").upper()
    # normalize common forms
    if "CEO" in t: return "CEO"
    if "CFO" in t: return "CFO"
    if "COO" in t: return "COO"
    if "PRES" in t or "PRESIDENT" in t: return "PRESIDENT"
    if "CHAIR" in t or "COB" in t: return "CHAIR"
    if "DIRECTOR" in t or t == "DIR" or t == "D": return "DIRECTOR"
    if "10" in t and "%" in t: return "10%"
    if "TEN" in t and "OWN" in t: return "10%"
    if "VICE" in t or "VP" in t: return "VP"
    if "GENERAL COUNSEL" in t or t == "GC": return "GENERAL COUNSEL"
    if "OFFICER" in t: return "OFFICER"
    if "OWNER" in t: return "OWNER"
    return "OTHER"

def rollup_by_ticker(
    df: pd.DataFrame,
    title_weights: Dict[str, float] = None,
    own_mode: str = "sum_pos",
    cluster_days: int = 7,
) -> pd.DataFrame:
    """
    Aggregate per-insider rows to per-ticker stats usable by the scoring step.
    own_mode: 'sum_pos' (default), 'mean_abs', or 'mean_signed' for OwnershipChangePct
    cluster_days: count trades within this window of the last TradeDate
    """
    if df is None or df.empty:
        return df

    gcols = ["Ticker", "Company", "Industry"]
    for c in gcols:
        if c not in df.columns:
            df[c] = None

    # Vectorized date parsing
    for col in ("TradeDate", "FilingDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str).str[:10], errors="coerce")

    # title weights
    tw = {**TITLE_MAP_DEFAULT, **(title_weights or {})}
    if "Title" in df.columns:
        df["_TitleKey"] = df["Title"].apply(_title_key)
        df["_TitleW"] = df["_TitleKey"].map(lambda k: tw.get(k, TITLE_MAP_DEFAULT.get(k, 0.30)))
    else:
        df["_TitleW"] = 0.50  # neutral default

    # ownership change
    if "OwnershipChangePct" not in df.columns:
        df["OwnershipChangePct"] = None

    def _agg_own(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return 0.0
        if own_mode == "sum_pos":
            return float(s[s > 0].sum())
        if own_mode == "mean_abs":
            return float(s.abs().mean())
        if own_mode == "mean_signed":
            return float(s.mean())
        return float(s[s > 0].sum())

    # cluster count within window of latest trade for the ticker
    def _cluster_count(sub: pd.DataFrame) -> int:
        if "TradeDate" not in sub or sub["TradeDate"].dropna().empty:
            return int(sub.shape[0])
        last = sub["TradeDate"].max()
        window = last - pd.Timedelta(days=int(cluster_days))
        return int(sub.loc[sub["TradeDate"] >= window].shape[0])

    # distinct insiders if available
    def _n_insiders(sub: pd.DataFrame) -> int:
        if "Insider" in sub.columns and sub["Insider"].notna().any():
            return int(sub["Insider"].nunique())
        if "NumInsiders" in sub.columns and sub["NumInsiders"].notna().any():
            return int(sub["NumInsiders"].max())
        return int(sub.shape[0])

    grouped = df.groupby(gcols, dropna=False)
    out = grouped.apply(
        lambda sub: pd.Series(
            {
                "TradeCount": int(sub.shape[0]),
                "DistinctInsiders": _n_insiders(sub),
                "TitleWeightedCount": float(sub["_TitleW"].sum()),
                "OwnershipChangeAgg": _agg_own(sub["OwnershipChangePct"]),
                "LastTradeDate": sub["TradeDate"].max() if "TradeDate" in sub else pd.NaT,
                "LastFilingDate": sub["FilingDate"].max() if "FilingDate" in sub else pd.NaT,
                "ClusterCount": _cluster_count(sub),
                "LatestTradePrice": pd.to_numeric(sub["TradePrice"], errors="coerce").dropna().iloc[-1]
                if "TradePrice" in sub and sub["TradePrice"].notna().any()
                else None,
                "TotalValueUSD": pd.to_numeric(sub["ValueUSD"], errors="coerce").dropna().sum()
                if "ValueUSD" in sub
                else None,
                "TotalQty": pd.to_numeric(sub["Qty"], errors="coerce").dropna().sum()
                if "Qty" in sub
                else None,
            }
        )
    ).reset_index()

    # days since last trade / filing
    now = pd.Timestamp.utcnow().normalize()
    if "LastTradeDate" in out.columns:
        out["DaysSinceTrade"] = (now - out["LastTradeDate"]).dt.days
    if "LastFilingDate" in out.columns:
        out["DaysSinceFiling"] = (now - out["LastFilingDate"]).dt.days

    return out

def scrape_and_rollup(
    base_url: str,
    pages: int = 1,
    sleep_sec: float = 0.7,
    title_weights: Dict[str, float] = None,
    own_mode: str = "sum_pos",
    cluster_days: int = 7,
) -> pd.DataFrame:
    """
    Convenience: scrape, then roll-up to per-ticker features in one call.
    """
    raw = scrape_openinsider(base_url, pages=pages, sleep_sec=sleep_sec)
    if raw.empty:
        return raw
    return rollup_by_ticker(
        raw,
        title_weights=title_weights,
        own_mode=own_mode,
        cluster_days=cluster_days,
    )
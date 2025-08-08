import ast
import json
import streamlit as st
import pandas as pd
import numpy as np

from openinsider_scraper import scrape_openinsider, scrape_openinsider_selenium, scrape_and_rollup
from insider_enrich import enrich_market_data
from insider_scoring import score, DEFAULT_WEIGHTS, DEFAULT_TITLE_WEIGHTS

# ---------- Page setup ----------
st.set_page_config(page_title="Automated OpenInsider Evaluator", layout="wide")

# ---------- Session state (fixes: "snaps back to upload") ----------
if "df" not in st.session_state:
    st.session_state.df = None
if "ranked" not in st.session_state:
    st.session_state.ranked = None
if "tw" not in st.session_state:
    st.session_state.tw = DEFAULT_TITLE_WEIGHTS.copy()
if "weights" not in st.session_state:
    st.session_state.weights = DEFAULT_WEIGHTS.copy()

st.title("Automated OpenInsider Evaluator (Scraping + yfinance)")

with st.expander("Instructions", expanded=False):
    st.write(
        """
        **Workflow**
        1) Paste an OpenInsider listing/screener URL (page 1). e.g. https://openinsider.com/screener  
        2) Click **Scrape** (requests/BS4) or **Scrape (Selenium)** if overlays/ads block content.  
        3) Optional: **Roll-up** to per-ticker features.  
        4) Click **Enrich** to add market cap & current price (yfinance).  
        5) Tune weights and **Rank Now**.  

        Tip: This app uses *session state* so your dataset persists across reruns—no more snapping back to the uploader.
        """
    )

# ---------- Controls to load data ----------
with st.container():
    st.subheader("Load data")
    url = st.text_input("OpenInsider URL (page 1)", value="", placeholder="https://openinsider.com/screener", key="url")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        pages = st.number_input("Pages", 1, 100, 3, 1, key="pages")
    with c2:
        delay = st.number_input("Delay between pages (sec)", 0.0, 5.0, 0.7, 0.1, key="delay")
    with c3:
        go = st.button("Scrape", use_container_width=True, key="btn_scrape")
    with c4:
        go_sel = st.button("Scrape (Selenium)", use_container_width=True, key="btn_scrape_sel")
    with c5:
        do_rollup = st.button("Roll-up by Ticker", use_container_width=True, key="btn_rollup")

    up_col1, up_col2 = st.columns([3,1])
    with up_col1:
        uploaded = st.file_uploader("Or upload CSV/Excel", type=["csv","xlsx"], key="uploader")
    with up_col2:
        use_uploaded = st.button("Use Uploaded", key="btn_use_upload")

# ---------- Actions: scrape / use upload ----------
def _set_df(df: pd.DataFrame, msg_ok: str):
    if df is None or df.empty:
        st.warning("No rows parsed.")
        return
    st.session_state.df = df
    st.session_state.ranked = None
    st.success(msg_ok + f" Rows: {len(df)}")

if go and url:
    with st.spinner("Scraping (requests + BeautifulSoup, pooled connections)..."):
        try:
            df = scrape_openinsider(url, pages=int(pages), sleep_sec=float(delay))
            _set_df(df, "Scrape complete.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Scrape error: {e}")

if go_sel and url:
    with st.spinner("Scraping (Selenium, headless)..."):
        try:
            df = scrape_openinsider_selenium(url, pages=int(pages), sleep_sec=float(delay))
            _set_df(df, "Selenium scrape complete.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Selenium scrape error: {e}")

if do_rollup:
    if st.session_state.df is None or st.session_state.df.empty:
        st.warning("Load data first, then roll-up.")
    else:
        with st.spinner("Aggregating to per-ticker features..."):
            from openinsider_scraper import rollup_by_ticker
            rolled = rollup_by_ticker(st.session_state.df)
            _set_df(rolled, "Rolled up.")
            st.experimental_rerun()

if use_uploaded and uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
        _set_df(df, "Upload loaded.")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Upload parse error: {e}")

# ---------- Main workspace ----------
if st.session_state.df is not None and not st.session_state.df.empty:
    st.subheader("Dataset")
    st.caption("This dataset persists in session state; interactions won't reset the page.")
    st.dataframe(st.session_state.df.head(200), use_container_width=True)

    act1, act2, act3 = st.columns([1,1,1])
    with act1:
        if st.button("Enrich with yfinance (batched)", key="btn_enrich"):
            with st.spinner("Fetching market caps & prices (batched requests)..."):
                st.session_state.df = enrich_market_data(st.session_state.df)
                st.success("Enriched with MarketCap, CurrentPrice, PriceDiffPct, TimeSinceTradeDays.")
                st.experimental_rerun()
    with act2:
        st.download_button("Download current dataset (CSV)",
                           st.session_state.df.to_csv(index=False).encode("utf-8"),
                           "openinsider_dataset.csv", "text/csv")
    with act3:
        if st.button("Reset dataset", key="btn_reset"):
            st.session_state.df = None
            st.session_state.ranked = None
            st.experimental_rerun()

    # ---------- Sidebar: weights ----------
    st.sidebar.header("Weights (0–1)")
    w = {}
    for k, default in st.session_state.weights.items():
        w[k] = st.sidebar.slider(k.replace("_"," ").title(), 0.0, 1.0, float(default), 0.01, key=f"w_{k}")
    st.session_state.weights = w

    st.sidebar.header("Title Weights")
    tw_default = st.session_state.tw
    tw_text = st.sidebar.text_area(
        "Edit title->weight mapping (JSON or Python dict)",
        value=json.dumps(tw_default, indent=2),
        height=220,
        key="tw_text"
    )
    TW = tw_default
    if tw_text:
        try:
            # Safe parsing: try JSON first, then literal_eval (no eval).
            try:
                TW = json.loads(tw_text)
            except json.JSONDecodeError:
                TW = ast.literal_eval(tw_text)
            if not isinstance(TW, dict):
                raise ValueError("Mapping must be a dict")
            st.session_state.tw = TW
        except Exception as ex:
            st.sidebar.warning(f"Parse error; keeping previous mapping. ({ex})")

    st.sidebar.header("Timing bonus")
    timing_days = st.sidebar.number_input("Bonus if trade within N days", 0, 30, 2, 1, key="timing_days")
    timing_mult = st.sidebar.number_input("Bonus multiplier", 1.0, 3.0, 1.10, 0.01, key="timing_mult")

    if st.button("Rank Now", key="btn_rank"):
        with st.spinner("Scoring & ranking..."):
            ranked = score(
                st.session_state.df,
                weights=st.session_state.weights,
                title_weights=st.session_state.tw,
                timing_bonus_days=int(timing_days),
                timing_bonus_mult=float(timing_mult)
            )
            st.session_state.ranked = ranked
            st.success("Ranking complete. See below.")
            st.experimental_rerun()

    if st.session_state.ranked is not None:
        st.subheader("Ranked Stocks")
        st.dataframe(st.session_state.ranked, use_container_width=True)
        st.download_button("Download ranked CSV",
                           st.session_state.ranked.to_csv(index=False).encode("utf-8"),
                           "ranked_insider_scores.csv", "text/csv")
        st.caption("Notes: Inputs are percentile-normalized; MarketCap uses inverse percentile (smaller = higher). Title multiplier averaged per ticker.")

else:
    st.info("Provide a URL or upload a file to begin. Once loaded, the dataset persists until you click **Reset dataset**.")
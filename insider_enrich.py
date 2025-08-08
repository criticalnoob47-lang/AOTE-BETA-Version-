
import pandas as pd
import numpy as np
import yfinance as yf

def enrich_market_data(df: pd.DataFrame, price_field: str = "TradePrice") -> pd.DataFrame:
    """
    Adds: MarketCap, CurrentPrice, PriceDiffPct (current vs trade price),
    TimeSinceTradeDays (by FilingDate if available, else TradeDate).

    Time handling is timezone-safe: all anchors are converted to UTC to avoid
    "tz-naive vs tz-aware" subtraction errors.
    """
    out = df.copy()

    # Ensure we have a trade price column
    if "TradePrice" not in out.columns and "LatestTradePrice" in out.columns:
        out["TradePrice"] = out["LatestTradePrice"]

    # Collect tickers
    if "Ticker" not in out.columns:
        out["Ticker"] = None
    tickers = sorted(pd.Series(out["Ticker"]).dropna().unique())

    # Fetch per-ticker market cap + last close
    caps = {}
    prices = {}

    for t in tickers:
        try:
            tk = yf.Ticker(t)
            # Market cap
            info = getattr(tk, "fast_info", None) or {}
            market_cap = getattr(info, "market_cap", None) if not isinstance(info, dict) else info.get("market_cap")
            if market_cap is None:
                # fallback to slower .info
                inf2 = tk.info or {}
                market_cap = inf2.get("marketCap")
            caps[t] = market_cap

            # Current price: last close from short history
            hist = tk.history(period="5d")
            if not hist.empty:
                prices[t] = float(hist["Close"].iloc[-1])
        except Exception:
            # best-effort; skip failures
            continue

    out["MarketCap"] = out["Ticker"].map(caps)
    out["CurrentPrice"] = out["Ticker"].map(prices)

    # PriceDiffPct (cheaper vs insider price is negative; caller may invert as needed)
    out["TradePrice"] = pd.to_numeric(out.get("TradePrice"), errors="coerce")
    out["PriceDiffPct"] = np.where(
        out["TradePrice"] > 0,
        (out["CurrentPrice"] - out["TradePrice"]) / out["TradePrice"],
        np.nan,
    )

    # ----- TimeSinceTradeDays (timezone-safe) -----
    # Prefer numeric columns if they already exist
    if "DaysSinceFiling" in out.columns:
        out["TimeSinceTradeDays"] = pd.to_numeric(out["DaysSinceFiling"], errors="coerce")
    elif "DaysSinceTrade" in out.columns:
        out["TimeSinceTradeDays"] = pd.to_numeric(out["DaysSinceTrade"], errors="coerce")
    else:
        # Parse to UTC to avoid tz-naive/aware subtraction errors
        for col in ["FilingDate", "TradeDate"]:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)

        # Build anchor: prefer filing date, otherwise trade date
        anchor = None
        if "FilingDate" in out.columns:
            anchor = out["FilingDate"]
        if anchor is None or anchor.isna().all():
            anchor = out.get("TradeDate")

        if anchor is not None:
            # Ensure series is UTC
            anchor = pd.to_datetime(anchor, errors="coerce", utc=True)
            now_utc = pd.Timestamp.now(tz="UTC").normalize()
            out["TimeSinceTradeDays"] = (now_utc - anchor).dt.days
        else:
            out["TimeSinceTradeDays"] = np.nan

    return out

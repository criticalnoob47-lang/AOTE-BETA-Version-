
import pandas as pd
import numpy as np

def percentile(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mask = s.notna()
    r = s[mask].rank(pct=True, method="average")
    out = pd.Series(0.0, index=s.index, dtype=float)
    out.loc[mask] = r
    return out

DEFAULT_TITLE_WEIGHTS = {
    "CEO": 1.00,
    "CFO": 0.95,
    "COO": 0.90,
    "PRESIDENT": 0.90,
    "CHAIR": 0.90,
    "DIRECTOR": 0.75,
    "10% OWNER": 0.60,
    "OFFICER": 0.50,
    "EXECUTIVE": 0.50,
    "VP": 0.50,
    "UNKNOWN": 0.30,
    "OTHER": 0.30
}

# New defaults prioritize rolled-up features by default (legacy kept but weight=0 to avoid double counting)
DEFAULT_WEIGHTS = {
    "title_weighted_count": 0.90,   # NEW: uses TitleWeightedCount if present
    "num_trades": 0.10,             # legacy fallback; low by default
    "ownership_agg": 0.85,          # NEW: uses OwnershipChangeAgg if present
    "ownership_change": 0.00,       # legacy fallback; off by default
    "cluster_count": 0.60,          # NEW
    "market_cap": 0.60,             # inverse percentile (smaller -> higher)
    "time_since_trade": 0.75,       # more recent -> higher
    "price_diff": 0.70              # cheaper vs insider price -> higher
}

def title_multiplier(title: str, mapping: dict) -> float:
    if not isinstance(title, str):
        return mapping.get("UNKNOWN", 0.30)
    T = title.upper()
    best = None
    for k, v in mapping.items():
        if k in T:
            best = v if best is None else max(best, v)
    if best is None:
        best = mapping.get("UNKNOWN", 0.30)
    return float(best)

def _aggregate_from_trades(x: pd.DataFrame, TW: dict, groupby: str, cluster_days: int):
    """Aggregate trade-level rows to per-ticker features matching the rolled-up schema."""
    x = x.copy()

    # Clean/prepare
    x["OwnershipChangePct"] = pd.to_numeric(x.get("OwnershipChangePct"), errors="coerce")
    x["OwnershipChangePct"] = x["OwnershipChangePct"].where(x["OwnershipChangePct"].notna(), 0.0)

    for col in ["FilingDate", "TradeDate"]:
        if col in x.columns:
            x[col] = pd.to_datetime(x[col], errors="coerce", utc=True)

    # Title multiplier per trade, then sum as TitleWeightedCount
    if "Title" in x.columns:
        x["TitleMult"] = x["Title"].apply(lambda t: title_multiplier(t, TW))
    else:
        x["TitleMult"] = 0.5  # neutral default if missing

    def _cluster(sub: pd.DataFrame) -> int:
        if "TradeDate" not in sub or sub["TradeDate"].dropna().empty:
            return int(sub.shape[0])
        last = sub["TradeDate"].max()
        window = last - pd.Timedelta(days=int(cluster_days))
        return int(sub.loc[sub["TradeDate"] >= window].shape[0])

    agg = x.groupby(groupby).agg(
        TradeCount = ("Insider", "count") if "Insider" in x.columns else (groupby, "size"),
        DistinctInsiders = ("Insider", "nunique") if "Insider" in x.columns else (groupby, "size"),
        TitleWeightedCount = ("TitleMult", "sum"),
        OwnershipChangeAgg = ("OwnershipChangePct", lambda s: float(s[s>0].sum())),
        latest_trade = ("TradeDate", "max") if "TradeDate" in x.columns else (groupby, "size"),
        latest_filing = ("FilingDate", "max") if "FilingDate" in x.columns else (groupby, "size"),
        TradePrice = ("TradePrice", "last") if "TradePrice" in x.columns else (groupby, "size"),
        MarketCap = ("MarketCap", "last") if "MarketCap" in x.columns else (groupby, "size"),
        PriceDiffPct = ("PriceDiffPct", "mean") if "PriceDiffPct" in x.columns else (groupby, "size"),
    ).reset_index()

    # Cluster count
    agg["ClusterCount"] = x.groupby(groupby).apply(_cluster).values

    # Time since trade (days) prefer FilingDate anchor if available
    now_utc = pd.Timestamp.now(tz="UTC").normalize()
    anchor = None
    if "latest_filing" in agg.columns:
        anchor = agg["latest_filing"]
    if anchor is None or anchor.isna().all():
        anchor = agg.get("latest_trade")
    if anchor is not None is not None:
        anchor = pd.to_datetime(anchor, errors="coerce", utc=True)
        agg["recent_days"] = (now_utc - anchor).dt.days
    else:
        agg["recent_days"] = np.nan

    # Keep consistent column names with rolled-up schema
    agg.rename(columns={
        "latest_trade": "LastTradeDate",
        "latest_filing": "LastFilingDate"
    }, inplace=True)

    return agg

def score(df: pd.DataFrame,
          weights: dict = None,
          title_weights: dict = None,
          groupby: str = "Ticker",
          cluster_days: int = 7,
          timing_bonus_days: int = 2,
          timing_bonus_mult: float = 1.10) -> pd.DataFrame:
    """
    Compute weighted scores at the stock level.
    Accepts either trade-level data OR a rolled-up dataset with columns like
    TitleWeightedCount, OwnershipChangeAgg, ClusterCount, DaysSinceTrade, etc.
    """
    W = {**DEFAULT_WEIGHTS, **(weights or {})}
    TW = {**DEFAULT_TITLE_WEIGHTS, **(title_weights or {})}

    x = df.copy()

    # Detect if data is already rolled up (no 'Insider' column but has rollup columns)
    pre_rolled = ("TitleWeightedCount" in x.columns) or ("TradeCount" in x.columns) or ("OwnershipChangeAgg" in x.columns)

    if pre_rolled:
        # Use provided features directly; make sure expected columns exist
        agg = x.copy()
        # Harmonize time columns for recency
        if "DaysSinceFiling" in agg.columns:
            agg["recent_days"] = pd.to_numeric(agg["DaysSinceFiling"], errors="coerce")
        elif "DaysSinceTrade" in agg.columns:
            agg["recent_days"] = pd.to_numeric(agg["DaysSinceTrade"], errors="coerce")
        else:
            # Try to compute from dates if present (timezone-safe)
            for col in ["LastFilingDate", "LastTradeDate"]:
                if col in agg.columns:
                    agg[col] = pd.to_datetime(agg[col], errors="coerce", utc=True)
            anchor = agg.get("LastFilingDate")
            if anchor is None or anchor.isna().all():
                anchor = agg.get("LastTradeDate")
            if anchor is not None:
                now_utc = pd.Timestamp.now(tz="UTC").normalize()
                agg["recent_days"] = (now_utc - anchor).dt.days
            else:
                agg["recent_days"] = np.nan
    else:
        # Aggregate from trade-level rows
        agg = _aggregate_from_trades(x, TW, groupby, cluster_days)

    # Market cap & price fields
    agg["MarketCap"] = pd.to_numeric(agg.get("MarketCap"), errors="coerce")
    agg["PriceDiffPct"] = pd.to_numeric(agg.get("PriceDiffPct"), errors="coerce")

    # Percentile transforms
    if "TitleWeightedCount" in agg.columns:
        agg["p_twc"] = percentile(agg["TitleWeightedCount"])
    else:
        agg["p_twc"] = 0.0

    if "TradeCount" in agg.columns:
        agg["p_num_trades"] = percentile(agg["TradeCount"])
    elif "DistinctInsiders" in agg.columns:
        agg["p_num_trades"] = percentile(agg["DistinctInsiders"])
    else:
        agg["p_num_trades"] = 0.0

    if "OwnershipChangeAgg" in agg.columns:
        agg["p_ownagg"] = percentile(agg["OwnershipChangeAgg"])
    else:
        agg["p_ownagg"] = 0.0

    if "ownchg_sum" in agg.columns:
        agg["p_ownchg"] = percentile(agg["ownchg_sum"])
    elif "OwnershipChangePct" in agg.columns:
        agg["p_ownchg"] = percentile(pd.to_numeric(agg["OwnershipChangePct"], errors="coerce"))
    else:
        agg["p_ownchg"] = 0.0

    if "ClusterCount" in agg.columns:
        agg["p_cluster"] = percentile(agg["ClusterCount"])
    else:
        agg["p_cluster"] = 0.0

    cap_pct = percentile(agg["MarketCap"])
    agg["p_mcap_inv"] = 1 - cap_pct

    agg["recent_days"] = pd.to_numeric(agg.get("recent_days"), errors="coerce")
    agg["p_recent"] = 1 - percentile(agg["recent_days"])

    cheapness = -agg["PriceDiffPct"]
    agg["p_price_rel"] = percentile(cheapness)

    agg["comp_twc"]      = W["title_weighted_count"] * agg["p_twc"]
    agg["comp_trades"]   = W["num_trades"] * agg["p_num_trades"]
    agg["comp_ownagg"]   = W["ownership_agg"] * agg["p_ownagg"]
    agg["comp_ownchg"]   = W["ownership_change"] * agg["p_ownchg"]
    agg["comp_cluster"]  = W["cluster_count"] * agg["p_cluster"]
    agg["comp_mcap"]     = W["market_cap"] * agg["p_mcap_inv"]
    agg["comp_time"]     = W["time_since_trade"] * agg["p_recent"]
    agg["comp_price"]    = W["price_diff"] * agg["p_price_rel"]

    agg["total_score"] = agg[[
        "comp_twc","comp_trades","comp_ownagg","comp_ownchg",
        "comp_cluster","comp_mcap","comp_time","comp_price"
    ]].sum(axis=1)

    if "recent_days" in agg.columns:
        bonus_mask = agg["recent_days"].le(timing_bonus_days)
        agg.loc[bonus_mask, "total_score"] *= float(timing_bonus_mult)

    agg = agg.sort_values("total_score", ascending=False).reset_index(drop=True)
    agg.insert(0, "Rank", agg.index + 1)
    return agg

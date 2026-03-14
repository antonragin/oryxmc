#!/usr/bin/env python3
"""Build historical_returns.json with NNLS regression backfill.

Downloads monthly and weekly return data from Quantum Axis and Yahoo Finance,
runs NNLS regression to backfill indices with incomplete history, and writes
the final JSON consumed by the OryxMC simulation engine.

Usage:
    python tools/build_data.py            # full rebuild
    python tools/build_data.py --dry-run  # download + regression, don't write JSON
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

# Add quantum client to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "quantum"))
import quantum_client  # noqa: E402

# ---------------------------------------------------------------------------
# Target window
# ---------------------------------------------------------------------------
TARGET_START = "2006-03"
TARGET_END = "2026-02"
TARGET_MONTHS = 240

# ---------------------------------------------------------------------------
# Index definitions
# ---------------------------------------------------------------------------
QUANTUM_INDICES = {
    "CDI":      {"quantum_name": "CDI",       "category": "br_fixed_income",
                 "name": "CDI",               "desc": "Taxa de Depósito Interbancário"},
    "IMA-B":    {"quantum_name": "IMA-B",     "category": "br_fixed_income",
                 "name": "IMA-B",             "desc": "Índice de Mercado ANBIMA - NTN-Bs"},
    "IMA-B 5":  {"quantum_name": "IMA-B 5",   "category": "br_fixed_income",
                 "name": "IMA-B 5",           "desc": "IMA-B até 5 anos"},
    "IMA-B 5+": {"quantum_name": "IMA-B 5+",  "category": "br_fixed_income",
                 "name": "IMA-B 5+",          "desc": "IMA-B acima de 5 anos"},
    "IRF-M":    {"quantum_name": "IRF-M",     "category": "br_fixed_income",
                 "name": "IRF-M",             "desc": "Índice de Renda Fixa de Mercado - LTN/NTN-F"},
    "IDA-DI":   {"quantum_name": "IDA-DI",    "category": "br_fixed_income",
                 "name": "TEVA-DI / IDA-DI",  "desc": "Índice Teva Debêntures DI (desde 2016-06), IDA-DI antes",
                 "teva_quantum_name": "Índice Teva Debêntures DI"},
    "IDA-IPCA": {"quantum_name": "IDA-IPCA",  "category": "br_fixed_income",
                 "name": "IDA-IPCA",          "desc": "Índice de Debêntures ANBIMA - IPCA"},
    "IDA-Infra": {"quantum_name": "IDA-IPCA Infraestrutura", "category": "br_fixed_income",
                  "name": "IDA-IPCA Infraestrutura", "desc": "Debêntures de Infraestrutura indexadas ao IPCA"},
    "Ibovespa": {"quantum_name": "Ibovespa",  "category": "br_equities",
                 "name": "Ibovespa",          "desc": "Índice Bovespa"},
    "IBX":      {"quantum_name": "IBX",       "category": "br_equities",
                 "name": "IBrX 100",          "desc": "Índice Brasil 100"},
    "IDIV":     {"quantum_name": "IDIV",      "category": "br_equities",
                 "name": "IDIV",              "desc": "Índice Dividendos BM&F Bovespa"},
    "IFIX":     {"quantum_name": "IFIX",      "category": "br_reits",
                 "name": "IFIX",              "desc": "Índice de Fundos Imobiliários"},
    "SP500TR":  {"quantum_name": "S&P 500 Total Return", "category": "intl_equities",
                 "name": "S&P 500 Total Return", "desc": "Índice S&P 500 retorno total em BRL"},
    "EEM":      {"quantum_name": "EEM",       "category": "intl_equities",
                 "name": "MSCI Emerging Markets", "desc": "Índice MSCI Mercados Emergentes em BRL"},
    "URTH":     {"quantum_name": "URTH",      "category": "intl_equities",
                 "name": "MSCI World",        "desc": "Índice MSCI World em BRL"},
    "ACWI":     {"quantum_name": "ACWI",      "category": "intl_equities",
                 "name": "MSCI ACWI",         "desc": "Índice MSCI All Country World em BRL"},
    "AGG":      {"quantum_name": "AGG",       "category": "intl_fixed_income",
                 "name": "Bloomberg US Agg Bond", "desc": "Índice Bloomberg US Aggregate Bond em BRL"},
    "SHY":      {"quantum_name": "SHY",       "category": "intl_fixed_income",
                 "name": "US Treasury 1-3 Anos", "desc": "Índice Barclays US Treasury 1-3 Year em BRL"},
    "TLT":      {"quantum_name": "TLT",       "category": "intl_fixed_income",
                 "name": "US Treasury 20+ Anos", "desc": "Índice Barclays US Treasury 20+ Year em BRL"},
    "VNQ":      {"quantum_name": "VNQ",       "category": "intl_reits",
                 "name": "US REITs (MSCI)",   "desc": "Índice MSCI US REIT em BRL"},
}

# Yahoo-only indices (monthly from Yahoo, not available on Quantum as long series)
YAHOO_INDICES = {
    "HYG": {"ticker": "HYG", "category": "intl_fixed_income",
            "name": "US High Yield Bond", "desc": "Índice iBoxx $ High Yield Corporate Bond em BRL"},
}

# International ETFs that need Yahoo monthly data to supplement Quantum
# (Quantum monthly starts only from ~2015 for these, but Yahoo has longer history)
YAHOO_MONTHLY_SUPPLEMENTS = ["EEM", "SHY", "TLT", "URTH", "ACWI", "AGG", "VNQ"]

# Yahoo tickers for weekly data supplements (regression)
YAHOO_WEEKLY_SUPPLEMENTS = {
    "TLT": "TLT", "SHY": "SHY", "EEM": "EEM",
    "URTH": "URTH", "ACWI": "ACWI", "HYG": "HYG",
    "AGG": "AGG", "VNQ": "VNQ",
}

USDBRL_TICKER = "BRL=X"

CATEGORIES = {
    "br_fixed_income": {"name": "Renda Fixa Brasil", "name_en": "BR Fixed Income"},
    "br_equities":     {"name": "Ações Brasil", "name_en": "BR Equities"},
    "intl_equities":   {"name": "Ações Internacionais", "name_en": "International Equities"},
    "intl_fixed_income": {"name": "Renda Fixa Internacional", "name_en": "International Fixed Income"},
    "intl_reits":      {"name": "REITs Internacionais", "name_en": "International REITs"},
    "br_reits":        {"name": "FIIs Brasil", "name_en": "BR REITs"},
}

# Anchor indices — those expected to have full 240-month history after merging
# Quantum + Yahoo monthly data
ANCHOR_INDICES = {
    "CDI", "IMA-B", "IMA-B 5", "IMA-B 5+", "IRF-M",
    "Ibovespa", "IBX", "IDIV",
    "SP500TR", "EEM", "AGG", "SHY", "TLT", "VNQ",
}

IPCA_QUANTUM_NAME = "IPCA"


# ---------------------------------------------------------------------------
# Data download helpers
# ---------------------------------------------------------------------------

def cumulative_to_periodic(series):
    """Convert cumulative % returns to periodic returns.

    Input:  [0.0, 1.5, 3.2, ...]  (cumulative % from date 0)
    Output: [0.015, 0.01674..., ...]  (periodic return per period)
    """
    cum = 1.0 + np.array(series) / 100.0
    periodic = cum[1:] / cum[:-1] - 1.0
    return periodic


def download_quantum_returns(client, name, periodicity="monthly"):
    """Download return series from Quantum and convert to periodic returns."""
    print(f"  Quantum {periodicity}: {name}...", end=" ", flush=True)
    data = client.get_returns(name, "since_inception", periodicity=periodicity)
    if not data:
        print("EMPTY")
        return pd.Series(dtype=float)

    dates = [d["date"] for d in data]
    values = [d["return_pct"] for d in data]
    periodic = cumulative_to_periodic(values)
    period_dates = dates[1:]

    if periodicity == "monthly":
        index = [d[:7] for d in period_dates]
    else:
        index = pd.to_datetime(period_dates)

    result = pd.Series(periodic, index=index, dtype=float)
    result = result[~result.index.duplicated(keep="first")]
    print(f"{len(result)} points")
    return result


def _yahoo_close(df, ticker):
    """Extract Close column from yfinance DataFrame handling MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        return df[("Close", ticker)]
    if "Close" in df.columns:
        return df["Close"]
    return df.iloc[:, 0]


def download_yahoo_weekly(tickers, start="2003-01-01"):
    """Download weekly adjusted close from Yahoo Finance, compute returns."""
    import yfinance as yf
    print(f"  Yahoo weekly: {tickers}...", end=" ", flush=True)
    ticker_list = tickers if isinstance(tickers, list) else [tickers]
    result = {}
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, start=start, interval="1wk",
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"{ticker}=EMPTY", end=" ")
                continue
            close = _yahoo_close(df, ticker)
            returns = close.pct_change().dropna()
            returns.index = pd.to_datetime(returns.index)
            result[ticker] = returns
        except Exception as e:
            print(f"{ticker}=ERROR({e})", end=" ")
    print(f"OK ({len(result)} tickers)")
    return result


def download_yahoo_monthly(tickers, start="2003-01-01"):
    """Download monthly adjusted close from Yahoo Finance, compute returns."""
    import yfinance as yf
    print(f"  Yahoo monthly: {tickers}...", end=" ", flush=True)
    ticker_list = tickers if isinstance(tickers, list) else [tickers]
    result = {}
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, start=start, interval="1mo",
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"{ticker}=EMPTY", end=" ")
                continue
            close = _yahoo_close(df, ticker)
            returns = close.pct_change().dropna()
            monthly = pd.Series(dtype=float)
            for dt, val in returns.items():
                key = f"{dt.year:04d}-{dt.month:02d}"
                monthly[key] = val
            monthly = monthly[~monthly.index.duplicated(keep="first")]
            result[ticker] = monthly
        except Exception as e:
            print(f"{ticker}=ERROR({e})", end=" ")
    print(f"OK ({len(result)} tickers)")
    return result


def download_usdbrl_weekly(start="2003-01-01"):
    """Download USDBRL weekly exchange rate returns from Yahoo Finance."""
    import yfinance as yf
    print("  Yahoo weekly: USDBRL...", end=" ", flush=True)
    df = yf.download(USDBRL_TICKER, start=start, interval="1wk",
                     auto_adjust=True, progress=False)
    if df.empty:
        print("EMPTY")
        return pd.Series(dtype=float)
    close = _yahoo_close(df, USDBRL_TICKER)
    fx_returns = close.pct_change().dropna()
    fx_returns.index = pd.to_datetime(fx_returns.index)
    print(f"{len(fx_returns)} points")
    return fx_returns


def download_usdbrl_monthly(start="2003-01-01"):
    """Download USDBRL monthly exchange rate returns from Yahoo Finance."""
    import yfinance as yf
    print("  Yahoo monthly: USDBRL...", end=" ", flush=True)
    df = yf.download(USDBRL_TICKER, start=start, interval="1mo",
                     auto_adjust=True, progress=False)
    if df.empty:
        print("EMPTY")
        return pd.Series(dtype=float)
    close = _yahoo_close(df, USDBRL_TICKER)
    fx_returns = close.pct_change().dropna()
    result = pd.Series(dtype=float)
    for dt, val in fx_returns.items():
        key = f"{dt.year:04d}-{dt.month:02d}"
        result[key] = val
    result = result[~result.index.duplicated(keep="first")]
    print(f"{len(result)} points")
    return result


def convert_to_brl_weekly(usd_returns, fx_returns):
    """Convert USD weekly returns to BRL: (1+r_usd)*(1+r_fx) - 1."""
    common = usd_returns.index.intersection(fx_returns.index)
    return (1 + usd_returns.loc[common]) * (1 + fx_returns.loc[common]) - 1


def convert_to_brl_monthly(usd_series, fx_series):
    """Convert USD monthly returns (YYYY-MM index) to BRL."""
    common = usd_series.index.intersection(fx_series.index)
    return (1 + usd_series.loc[common]) * (1 + fx_series.loc[common]) - 1


def to_isoweek(dt_index):
    """Convert DatetimeIndex to ISO year-week strings for alignment."""
    return pd.Index([f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}" for d in dt_index])


def align_weekly_to_isoweek(series):
    """Re-index a weekly DatetimeIndex series by ISO year-week.

    If multiple observations fall in the same ISO week, keep the first.
    """
    if series.empty:
        return pd.Series(dtype=float)
    weeks = to_isoweek(series.index)
    result = pd.Series(series.values, index=weeks, dtype=float)
    return result[~result.index.duplicated(keep="first")]


# ---------------------------------------------------------------------------
# NNLS regression backfill
# ---------------------------------------------------------------------------

def run_nnls_backfill(target_weekly, anchor_weekly_dict, target_monthly,
                      anchor_monthly_dict, target_start=TARGET_START,
                      exclude_self=None):
    """Run NNLS regression on weekly overlap and apply to monthly gaps.

    Weekly series are aligned by ISO week to handle date mismatches between
    Quantum and Yahoo Finance data sources.
    """
    if target_weekly.empty:
        raise ValueError("Target weekly series is empty")

    # Align all weekly series by ISO week
    target_wk = align_weekly_to_isoweek(target_weekly)
    anchor_wk = {}
    for name, series in anchor_weekly_dict.items():
        if name == exclude_self:
            continue
        aligned = align_weekly_to_isoweek(series)
        if not aligned.empty:
            anchor_wk[name] = aligned

    # Build regression matrix via inner join
    df = pd.DataFrame({"target": target_wk})
    for name, series in anchor_wk.items():
        df[name] = series
    df = df.dropna()

    if len(df) < 20:
        raise ValueError(f"Insufficient weekly overlap: {len(df)} rows (need >= 20)")

    y = df["target"].values
    anchor_names = [c for c in df.columns if c != "target"]
    X = df[anchor_names].values

    # Run NNLS
    coeffs, residual = nnls(X, y)

    # R-squared
    ss_res = np.sum((y - X @ coeffs) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    coeff_dict = {name: float(c) for name, c in zip(anchor_names, coeffs) if c > 1e-8}

    print(f"    NNLS: {len(df)} weekly obs, R²={r_squared:.4f}, "
          f"{len(coeff_dict)} active regressors")
    for name, c in sorted(coeff_dict.items(), key=lambda x: -x[1]):
        print(f"      {name}: {c:.4f}")

    # Identify missing months
    all_target_months = generate_month_range(target_start, TARGET_END)
    existing_months = set(target_monthly.index)
    missing_months = [m for m in all_target_months if m not in existing_months]

    # Build synthetic returns for missing months
    synthetic = {}
    for month in missing_months:
        ret = 0.0
        valid = True
        for name, c in coeff_dict.items():
            anchor_m = anchor_monthly_dict.get(name, pd.Series(dtype=float))
            if month in anchor_m.index:
                ret += c * anchor_m[month]
            else:
                valid = False
                break
        if valid:
            synthetic[month] = ret

    # Merge: synthetic + native
    full = {}
    for month in all_target_months:
        if month in existing_months:
            full[month] = float(target_monthly[month])
        elif month in synthetic:
            full[month] = synthetic[month]

    full_monthly = pd.Series(full).sort_index()

    return {
        "full_monthly": full_monthly,
        "coefficients": coeff_dict,
        "r_squared": r_squared,
        "synthetic_months": sorted(synthetic.keys()),
        "native_months": sorted(existing_months),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_month_range(start, end):
    """Generate list of YYYY-MM strings from start to end inclusive."""
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return months


# ---------------------------------------------------------------------------
# Main build orchestrator
# ---------------------------------------------------------------------------

def build(dry_run=False):
    print("=" * 60)
    print("OryxMC Data Build — NNLS Regression Backfill")
    print(f"Target window: {TARGET_START} to {TARGET_END} ({TARGET_MONTHS} months)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Connect to Quantum
    # ------------------------------------------------------------------
    print("\n[1/8] Connecting to Quantum Axis...")
    client = quantum_client.connect()
    print("  Connected.")

    # ------------------------------------------------------------------
    # 2. Download monthly returns from Quantum
    # ------------------------------------------------------------------
    print("\n[2/8] Downloading monthly returns from Quantum...")
    quantum_monthly = {}
    for key, info in QUANTUM_INDICES.items():
        series = download_quantum_returns(client, info["quantum_name"], "monthly")
        series = series[(series.index >= TARGET_START) & (series.index <= TARGET_END)]
        quantum_monthly[key] = series
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # 3. Download IPCA monthly from Quantum
    # ------------------------------------------------------------------
    print("\n[3/8] Downloading IPCA...")
    ipca_series = download_quantum_returns(client, IPCA_QUANTUM_NAME, "monthly")
    ipca_series = ipca_series[(ipca_series.index >= TARGET_START) &
                              (ipca_series.index <= TARGET_END)]
    print(f"  IPCA: {len(ipca_series)} months")

    # ------------------------------------------------------------------
    # 4. Download weekly returns for regression
    # ------------------------------------------------------------------
    print("\n[4/8] Downloading weekly returns for regression...")

    # Quantum weekly
    quantum_weekly = {}
    for key, info in QUANTUM_INDICES.items():
        series = download_quantum_returns(client, info["quantum_name"], "weekly")
        if not series.empty:
            quantum_weekly[key] = series
        time.sleep(0.5)

    # Yahoo weekly (USD)
    yahoo_weekly_usd = download_yahoo_weekly(list(YAHOO_WEEKLY_SUPPLEMENTS.values()))

    # USDBRL weekly + monthly
    usdbrl_weekly = download_usdbrl_weekly()
    usdbrl_monthly = download_usdbrl_monthly()

    # Convert Yahoo USD weekly to BRL
    yahoo_weekly_brl = {}
    for idx_key, ticker in YAHOO_WEEKLY_SUPPLEMENTS.items():
        if ticker in yahoo_weekly_usd:
            yahoo_weekly_brl[idx_key] = convert_to_brl_weekly(
                yahoo_weekly_usd[ticker], usdbrl_weekly
            )

    # Merge weekly: Quantum + Yahoo (Quantum preferred on exact date match)
    merged_weekly = {}
    all_idx_keys = set(list(QUANTUM_INDICES.keys()) + list(YAHOO_INDICES.keys()))
    for key in all_idx_keys:
        parts = []
        if key in quantum_weekly:
            parts.append(quantum_weekly[key])
        if key in yahoo_weekly_brl:
            parts.append(yahoo_weekly_brl[key])
        if parts:
            combined = pd.concat(parts)
            combined = combined[~combined.index.duplicated(keep="first")]
            merged_weekly[key] = combined.sort_index()

    # ------------------------------------------------------------------
    # 5. Download Yahoo monthly supplements + HYG, merge with Quantum
    # ------------------------------------------------------------------
    print("\n[5/8] Downloading Yahoo monthly supplements...")

    # All tickers we need Yahoo monthly for
    yahoo_monthly_tickers = list(YAHOO_INDICES.keys()) + YAHOO_MONTHLY_SUPPLEMENTS
    yahoo_monthly_tickers = list(dict.fromkeys(yahoo_monthly_tickers))  # dedup
    yahoo_monthly_usd = download_yahoo_monthly(yahoo_monthly_tickers)

    # Convert USD→BRL
    print("\n[6/8] Merging Quantum + Yahoo monthly data...")
    all_monthly = dict(quantum_monthly)  # start with Quantum data

    for key in yahoo_monthly_tickers:
        ticker = key  # ticker == key for these
        if key in YAHOO_INDICES:
            ticker = YAHOO_INDICES[key]["ticker"]

        if ticker not in yahoo_monthly_usd:
            continue

        yahoo_brl = convert_to_brl_monthly(yahoo_monthly_usd[ticker], usdbrl_monthly)
        yahoo_brl = yahoo_brl[(yahoo_brl.index >= TARGET_START) &
                              (yahoo_brl.index <= TARGET_END)]

        if key in all_monthly and not all_monthly[key].empty:
            # Merge: Quantum preferred, Yahoo fills gaps
            quantum_m = all_monthly[key]
            merged = quantum_m.copy()
            for month in yahoo_brl.index:
                if month not in merged.index:
                    merged[month] = yahoo_brl[month]
            all_monthly[key] = merged.sort_index()
            added = len(merged) - len(quantum_m)
            if added > 0:
                print(f"  {key}: {len(quantum_m)} Quantum + {added} Yahoo = {len(merged)} months")
        else:
            all_monthly[key] = yahoo_brl
            print(f"  {key}: {len(yahoo_brl)} months (Yahoo only)")

    # Merge TEVA Debêntures DI into IDA-DI (TEVA preferred where available)
    teva_name = QUANTUM_INDICES.get("IDA-DI", {}).get("teva_quantum_name")
    if teva_name:
        teva_series = download_quantum_returns(client, teva_name, "monthly")
        teva_series = teva_series[(teva_series.index >= TARGET_START) &
                                  (teva_series.index <= TARGET_END)]
        if not teva_series.empty and "IDA-DI" in all_monthly:
            ida_before = all_monthly["IDA-DI"].copy()
            # TEVA preferred, IDA-DI fills gaps
            merged = teva_series.copy()
            for m in ida_before.index:
                if m not in merged.index:
                    merged[m] = ida_before[m]
            all_monthly["IDA-DI"] = merged.sort_index()
            print(f"  IDA-DI: merged {len(teva_series)} TEVA + "
                  f"{len(merged) - len(teva_series)} IDA-DI = {len(merged)} months")

    # Filter all to target window
    for key in list(all_monthly.keys()):
        s = all_monthly[key]
        all_monthly[key] = s[(s.index >= TARGET_START) & (s.index <= TARGET_END)]

    # Print current state
    print("\n  Monthly data coverage:")
    for key in list(QUANTUM_INDICES.keys()) + list(YAHOO_INDICES.keys()):
        if key in all_monthly and not all_monthly[key].empty:
            s = all_monthly[key]
            print(f"    {key}: {len(s)} months ({s.index[0]} to {s.index[-1]})")

    # ------------------------------------------------------------------
    # 7. Run NNLS backfill for incomplete indices
    # ------------------------------------------------------------------
    print("\n[7/8] Running NNLS regression backfill...")

    needs_backfill = {}
    for key, series in all_monthly.items():
        missing = TARGET_MONTHS - len(series)
        if missing > 0:
            needs_backfill[key] = missing

    backfill_order = sorted(needs_backfill.keys(), key=lambda k: needs_backfill[k])

    anchors_ready = {k for k in ANCHOR_INDICES
                     if k in all_monthly and len(all_monthly[k]) >= TARGET_MONTHS}
    print(f"  Full anchors ({len(anchors_ready)}): {', '.join(sorted(anchors_ready))}")
    print(f"  Need backfill ({len(backfill_order)}):")
    for key in backfill_order:
        print(f"    {key}: {len(all_monthly[key])} months, {needs_backfill[key]} missing")

    # Build anchor dicts (only indices with full 240-month coverage)
    anchor_weekly = {k: merged_weekly[k] for k in anchors_ready if k in merged_weekly}
    anchor_monthly = {k: all_monthly[k] for k in anchors_ready}

    regression_metadata = {}
    for key in backfill_order:
        print(f"\n  Backfilling {key}...")
        if key not in merged_weekly:
            print(f"    WARNING: No weekly data for {key}, skipping backfill")
            continue

        try:
            result = run_nnls_backfill(
                target_weekly=merged_weekly[key],
                anchor_weekly_dict=anchor_weekly,
                target_monthly=all_monthly[key],
                anchor_monthly_dict=anchor_monthly,
                target_start=TARGET_START,
                exclude_self=key,  # don't regress against self
            )
            all_monthly[key] = result["full_monthly"]
            regression_metadata[key] = {
                "coefficients": result["coefficients"],
                "r_squared": round(result["r_squared"], 6),
                "synthetic_months": len(result["synthetic_months"]),
                "native_months": len(result["native_months"]),
                "synthetic_range": (result["synthetic_months"][0],
                                    result["synthetic_months"][-1])
                                   if result["synthetic_months"] else None,
            }
            print(f"    Result: {len(result['full_monthly'])} total months, "
                  f"{len(result['synthetic_months'])} synthetic")

            # If this was an anchor that now has 240 months, add to anchor pool
            if key in ANCHOR_INDICES and len(result["full_monthly"]) >= TARGET_MONTHS:
                anchor_monthly[key] = result["full_monthly"]
                print(f"    → Now available as anchor")
        except Exception as e:
            print(f"    ERROR: {e}")

    # ------------------------------------------------------------------
    # 8. Assemble and write JSON
    # ------------------------------------------------------------------
    print("\n[8/8] Assembling output JSON...")

    indices_out = {}
    all_keys = list(QUANTUM_INDICES.keys()) + list(YAHOO_INDICES.keys())

    for key in all_keys:
        info = QUANTUM_INDICES.get(key) or YAHOO_INDICES[key]

        if key not in all_monthly or all_monthly[key].empty:
            print(f"  WARNING: No data for {key}, skipping")
            continue

        series = all_monthly[key]
        series = series[(series.index >= TARGET_START) & (series.index <= TARGET_END)]
        months_sorted = sorted(series.index)

        returns_dict = {m: round(float(series[m]), 10) for m in months_sorted}

        indices_out[key] = {
            "category": info["category"],
            "name": info["name"],
            "desc": info["desc"],
            "months_available": len(returns_dict),
            "start_date": months_sorted[0],
            "end_date": months_sorted[-1],
            "returns": returns_dict,
        }

    ipca_dict = {m: round(float(ipca_series[m]), 10) for m in sorted(ipca_series.index)}

    output = {
        "metadata": {
            "collected_at": datetime.now().strftime("%Y-%m-%d"),
            "target_start": TARGET_START,
            "target_end": TARGET_END,
            "target_months": TARGET_MONTHS,
            "currency": "BRL",
            "note": "All returns in BRL. International returns converted using USDBRL monthly rate.",
            "regression_backfill": regression_metadata,
        },
        "categories": CATEGORIES,
        "ipca": ipca_dict,
        "indices": indices_out,
    }

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Indices: {len(indices_out)}")
    print(f"  IPCA months: {len(ipca_dict)}")
    for key, idx in indices_out.items():
        marker = " (backfilled)" if key in regression_metadata else ""
        print(f"  {key}: {idx['months_available']} months "
              f"({idx['start_date']} to {idx['end_date']}){marker}")

    # Validate
    print("\nValidating...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from engine import validate_data
    try:
        validate_data(output)
        print("  Validation PASSED")
    except Exception as e:
        print(f"  Validation FAILED: {e}")
        if not dry_run:
            print("  Aborting write.")
            return False

    if dry_run:
        print("\n  --dry-run: not writing JSON")
    else:
        out_path = Path(__file__).resolve().parent.parent / "data" / "historical_returns.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=None, separators=(",", ":"))
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"\n  Wrote {out_path} ({size_mb:.2f} MB)")

        # Compute efficient frontier (requires scipy, available at build time)
        print("\n[9/9] Computing efficient frontier...")
        from compute_frontier import compute_frontier as _compute_ef, compute_data_checksum
        frontier = _compute_ef(output)
        output["efficient_frontier"] = frontier
        with open(out_path, "w") as f:
            json.dump(output, f, indent=None, separators=(",", ":"))
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Rewrote {out_path} ({size_mb:.2f} MB) with frontier")
        print(f"  Data checksum: {frontier['data_checksum']}")

    print("\nDone.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build historical_returns.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Download and compute but don't write JSON")
    args = parser.parse_args()
    success = build(dry_run=args.dry_run)
    sys.exit(0 if success else 1)

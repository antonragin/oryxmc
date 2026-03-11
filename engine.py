"""Monte Carlo simulation engine for Brazilian portfolio simulator.

All returns are in BRL. Supports nominal and real (IPCA-adjusted) simulations
with optional periodic withdrawals.
"""
import json
import numpy as np
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "historical_returns.json"

CATEGORIES = {
    "br_fixed_income": "Renda Fixa Brasil",
    "br_equities": "Ações Brasil",
    "intl_equities": "Ações Internacionais",
    "intl_fixed_income": "Renda Fixa Internacional",
    "intl_reits": "REITs Internacionais",
}

# Fallback indices when no selected index in a category has data for a month
CATEGORY_FALLBACKS = {
    "br_fixed_income": "CDI",
    "br_equities": "Ibovespa",
    "intl_equities": "SP500TR",
    "intl_fixed_income": "AGG",
    "intl_reits": "VNQ",
}


def load_data():
    with open(DATA_FILE) as f:
        return json.load(f)


def _parse_month(month_str):
    if not isinstance(month_str, str) or len(month_str) != 7 or month_str[4] != "-":
        raise ValueError(f"Mês inválido: {month_str!r}")
    year = int(month_str[:4])
    month = int(month_str[5:7])
    if month < 1 or month > 12:
        raise ValueError(f"Mês inválido: {month_str!r}")
    return year, month


def _month_range(start_month, end_month):
    start = _parse_month(start_month)
    end = _parse_month(end_month)
    if start > end:
        raise ValueError(f"Janela inválida: {start_month} > {end_month}")
    months = []
    year, month = start
    while (year, month) <= end:
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def _validate_series(name, series):
    if not isinstance(series, dict) or not series:
        raise ValueError(f"{name}: série ausente ou vazia")
    months = sorted(series.keys())
    for month in months:
        _parse_month(month)
        value = series[month]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{name}: valor não-finito em {month}")
        if value <= -1:
            raise ValueError(f"{name}: valor <= -100% em {month}")
    expected = _month_range(months[0], months[-1])
    if months != expected:
        raise ValueError(f"{name}: meses faltando entre {months[0]} e {months[-1]}")
    return months


def validate_data(data):
    if not isinstance(data, dict):
        raise ValueError("Arquivo de dados inválido")
    metadata = data.get("metadata")
    ipca = data.get("ipca")
    indices = data.get("indices")
    if not isinstance(metadata, dict):
        raise ValueError("metadata ausente")
    if not isinstance(ipca, dict) or not ipca:
        raise ValueError("Série IPCA ausente")
    if not isinstance(indices, dict) or not indices:
        raise ValueError("Nenhum índice carregado")
    target_start = metadata.get("target_start")
    target_end = metadata.get("target_end")
    target_months = _month_range(target_start, target_end)
    _validate_series("IPCA", ipca)
    missing_ipca = [m for m in target_months if m not in ipca]
    if missing_ipca:
        raise ValueError(
            f"IPCA incompleto na janela-alvo: {missing_ipca[0]} ... {missing_ipca[-1]}"
        )
    # Validate every index structure/data first
    for key, info in indices.items():
        if not isinstance(info, dict):
            raise ValueError(f"Índice inválido: {key}")
        for field in ("name", "desc", "category", "returns", "start_date", "end_date", "months_available"):
            if field not in info:
                raise ValueError(f"{key}: campo ausente {field}")
        if not isinstance(info["name"], str) or not info["name"].strip():
            raise ValueError(f"{key}: name inválido")
        if isinstance(info["months_available"], bool) or not isinstance(info["months_available"], int):
            raise ValueError(f"{key}: months_available inválido")
        if info["category"] not in CATEGORIES:
            raise ValueError(f"{key}: categoria desconhecida {info['category']}")
        months = _validate_series(f"Índice {key}", info["returns"])
        if info["months_available"] != len(months):
            raise ValueError(
                f"{key}: months_available={info['months_available']} != {len(months)}"
            )
        if info["start_date"] != months[0] or info["end_date"] != months[-1]:
            raise ValueError(
                f"{key}: start/end inconsistentes ({info['start_date']} a {info['end_date']})"
            )
    # Then validate fallback coverage (now safe because all indices are validated)
    for cat, fallback in CATEGORY_FALLBACKS.items():
        if fallback not in indices:
            raise ValueError(f"Fallback ausente para {cat}: {fallback}")
        fb_info = indices[fallback]
        if fb_info["category"] != cat:
            raise ValueError(
                f"Fallback {fallback} pertence a {fb_info['category']}, esperado {cat}"
            )
        missing_fb = [m for m in target_months if m not in fb_info["returns"]]
        if missing_fb:
            raise ValueError(
                f"Fallback {fallback} incompleto: {missing_fb[0]} ... {missing_fb[-1]}"
            )


def get_available_indices(data):
    target_start = data["metadata"]["target_start"]
    target_end = data["metadata"]["target_end"]
    target_months_set = {m for m in data["ipca"] if target_start <= m <= target_end}
    target_months = len(target_months_set)
    indices = []
    for key, info in data["indices"].items():
        # Count months this index actually has within the target window
        months_in_window = sum(1 for m in info["returns"] if m in target_months_set)
        missing_in_window = target_months - months_in_window
        indices.append({
            "key": key,
            "name": info["name"],
            "desc": info["desc"],
            "category": info["category"],
            "category_name": CATEGORIES.get(info["category"], info["category"]),
            "months_available": info["months_available"],
            "months_in_window": months_in_window,
            "missing_months_in_window": missing_in_window,
            "coverage_pct": round(months_in_window / target_months * 100, 1) if target_months > 0 else 100.0,
            "start_date": info["start_date"],
            "end_date": info["end_date"],
            "has_full_history": months_in_window >= target_months,
        })
    return indices


def build_portfolio_returns(data, allocations):
    """
    Build the portfolio monthly return series applying the substitution rule.

    Substitution rule:
    - For months where a selected index has no data, redistribute its weight
      pro-rata among other selected indices in the same category that DO have data.
    - If no selected index in the category has data, use a predefined category
      fallback index (e.g., CDI for BR fixed income, Ibovespa for BR equities).
    """
    target_start = data["metadata"]["target_start"]
    target_end = data["metadata"]["target_end"]
    all_months = sorted(m for m in data["ipca"] if target_start <= m <= target_end)

    # Validate and normalize weights to exactly 1.0
    for k, v in allocations.items():
        if v < 0:
            raise ValueError(f"Alocação negativa não permitida: {k}")
    clean = {k: v for k, v in allocations.items() if v > 0}
    total_weight = sum(clean.values())
    if total_weight <= 0:
        raise ValueError("Alocação total deve ser > 0")
    allocations = {k: v / total_weight for k, v in clean.items()}

    # Group allocations by category
    cat_allocs = {}
    for idx_key, weight in allocations.items():
        cat = data["indices"][idx_key]["category"]
        cat_allocs.setdefault(cat, {})[idx_key] = weight

    portfolio_returns = []
    months_used = []

    # Track actual substitutions per missing index
    sub_log = {}
    for cat, idx_weights in cat_allocs.items():
        for idx_key in idx_weights:
            sub_log[idx_key] = {"peer_months": 0, "fallback_months": 0,
                                "peers": set(), "fallback": None}

    for month in all_months:
        month_return = 0.0

        for cat, idx_weights in cat_allocs.items():
            available = {}
            missing = {}
            for idx_key, weight in idx_weights.items():
                if month in data["indices"][idx_key]["returns"]:
                    available[idx_key] = weight
                else:
                    missing[idx_key] = weight

            missing_weight = sum(missing.values())

            if missing_weight > 0:
                if available:
                    # Redistribute pro-rata among available selected indices
                    total_avail = sum(available.values())
                    for k in list(available):
                        available[k] += missing_weight * (available[k] / total_avail)
                    for mk in missing:
                        sub_log[mk]["peer_months"] += 1
                        sub_log[mk]["peers"].update(available.keys())
                else:
                    # Use category fallback
                    fallback = CATEGORY_FALLBACKS.get(cat)
                    if fallback and month in data["indices"][fallback]["returns"]:
                        available = {fallback: sum(idx_weights.values())}
                        for mk in missing:
                            sub_log[mk]["fallback_months"] += 1
                            sub_log[mk]["fallback"] = fallback
                    else:
                        raise ValueError(
                            f"Sem substituto disponível para {CATEGORIES.get(cat, cat)} em {month}."
                        )

            for idx_key, eff_weight in available.items():
                month_return += eff_weight * data["indices"][idx_key]["returns"][month]

        portfolio_returns.append(month_return)
        months_used.append(month)

    # Generate warnings from actual substitution log
    warnings = []
    for idx_key, log in sub_log.items():
        total_missing = log["peer_months"] + log["fallback_months"]
        if total_missing == 0:
            continue
        idx_info = data["indices"][idx_key]
        weight = allocations[idx_key]
        parts = []
        if log["peer_months"] > 0:
            peer_names = ", ".join(data["indices"][k]["name"] for k in sorted(log["peers"]))
            parts.append(
                f"em {log['peer_months']} meses redistribuída para: {peer_names}"
            )
        if log["fallback_months"] > 0:
            fb_name = data["indices"][log["fallback"]]["name"]
            parts.append(
                f"em {log['fallback_months']} meses utilizou-se {fb_name} como proxy"
            )
        warnings.append(
            f"{idx_info['name']} ({weight*100:.1f}%) — histórico "
            f"{idx_info['start_date']} a {idx_info['end_date']}. "
            f"Substituição em {total_missing} meses: "
            + "; ".join(parts) + "."
        )

    # Align IPCA — raise error if missing
    ipca_values = []
    for m in months_used:
        if m not in data["ipca"]:
            raise ValueError(f"IPCA ausente para {m}")
        ipca_values.append(data["ipca"][m])

    return {
        "months": months_used,
        "portfolio_returns": np.array(portfolio_returns),
        "ipca": np.array(ipca_values),
        "warnings": warnings,
    }


def run_monte_carlo(portfolio_returns, ipca, initial_value, n_years,
                    n_trajectories=10000, withdrawal_annual=0.0, seed=None,
                    benchmark_returns=None, benchmark_name=None):
    """
    Run Monte Carlo simulation using bootstrap resampling.

    Withdrawals are applied monthly (1/12 of annual amount per month),
    adjusted for cumulative inflation in that trajectory.
    """
    if n_years <= 0:
        raise ValueError("Horizonte deve ser > 0")

    portfolio_returns = np.asarray(portfolio_returns, dtype=float)
    ipca = np.asarray(ipca, dtype=float)
    if portfolio_returns.ndim != 1 or ipca.ndim != 1:
        raise ValueError("Séries históricas devem ser vetores 1D")
    if portfolio_returns.size == 0 or ipca.size == 0:
        raise ValueError("Séries históricas vazias")
    if portfolio_returns.size != ipca.size:
        raise ValueError("Séries históricas desalinhadas")
    if not np.all(np.isfinite(portfolio_returns)) or not np.all(np.isfinite(ipca)):
        raise ValueError("Séries históricas contêm valores não-finitos")

    rng = np.random.default_rng(seed)

    n_months = n_years * 12
    n_hist = len(portfolio_returns)
    has_withdrawals = withdrawal_annual > 0
    monthly_withdrawal_real = withdrawal_annual / 12.0

    if benchmark_returns is not None:
        benchmark_returns = np.asarray(benchmark_returns, dtype=float)
        if benchmark_returns.ndim != 1 or benchmark_returns.size != n_hist:
            raise ValueError("Benchmark desalinhado com o histórico")
        if not np.all(np.isfinite(benchmark_returns)):
            raise ValueError("Benchmark contém valores não-finitos")

    # Bootstrap: sample month indices with replacement
    sampled_idx = rng.integers(0, n_hist, size=(n_trajectories, n_months))

    # IMPORTANT: On some numpy builds / platforms, operations like (1.0 + array),
    # np.cumsum, np.true_divide with out= etc. can silently mutate source arrays
    # for arrays exceeding certain size thresholds. We protect against this by
    # using .copy() liberally and avoiding patterns known to trigger the bug.
    sampled_returns = portfolio_returns[sampled_idx].copy()
    sampled_ipca = ipca[sampled_idx].copy()

    # Real returns: (1+r)/(1+i) - 1 = expm1(log1p(r) - log1p(i))
    log_ret = np.log1p(sampled_returns)    # does NOT modify sampled_returns
    log_ipca = np.log1p(sampled_ipca)      # does NOT modify sampled_ipca
    sampled_real_returns = np.expm1(log_ret - log_ipca)

    # Cumulative inflation: cumprod(1+i) = exp(cumsum(log1p(i)))
    # Use a COPY of log_ipca since cumsum may mutate the input on some builds
    cum_log_ipca = np.cumsum(log_ipca.copy(), axis=1)
    cum_inflation = np.ones((n_trajectories, n_months + 1), dtype=float)
    cum_inflation[:, 1:] = np.exp(cum_log_ipca)

    def simulate_nominal_paths(monthly_rets):
        """Build trajectory from monthly returns (not log returns)."""
        if not has_withdrawals:
            # Build cumulative product via log space using a fresh copy
            log_r = np.log1p(monthly_rets.copy())
            cum_log_r = np.cumsum(log_r, axis=1)
            traj = np.empty((n_trajectories, n_months + 1), dtype=float)
            traj[:, 0] = initial_value
            traj[:, 1:] = initial_value * np.exp(cum_log_r)
            return traj
        # Withdrawal path needs per-step computation
        traj = np.full((n_trajectories, n_months + 1), initial_value, dtype=float)
        for t in range(n_months):
            traj[:, t + 1] = traj[:, t] * (1.0 + monthly_rets[:, t])
            withdrawal = monthly_withdrawal_real * cum_inflation[:, t + 1]
            traj[:, t + 1] = np.maximum(traj[:, t + 1] - withdrawal, 0.0)
        return traj

    # Portfolio trajectories
    traj_nominal = simulate_nominal_paths(sampled_returns)
    traj_real = (traj_nominal.copy()) / cum_inflation

    # Benchmark trajectories (same sampled months)
    benchmark_nominal = None
    benchmark_real = None
    sampled_benchmark = None
    sampled_benchmark_real = None
    if benchmark_returns is not None:
        sampled_benchmark = benchmark_returns[sampled_idx].copy()
        log_bench = np.log1p(sampled_benchmark)
        sampled_benchmark_real = np.expm1(log_bench - log_ipca)
        benchmark_nominal = simulate_nominal_paths(sampled_benchmark)
        benchmark_real = (benchmark_nominal.copy()) / cum_inflation

    # Compute statistics
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    time_years = np.arange(n_months + 1) / 12.0

    def compute_stats(trajectories, sampled_rets, bench_trajectories=None, sampled_bench=None):
        final = trajectories[:, -1]
        pct_values = np.percentile(trajectories, percentiles, axis=0)
        pct_lines = {p: pct_values[i].tolist() for i, p in enumerate(percentiles)}

        valid = final > 0
        cagrs = np.full(final.shape, -1.0, dtype=float)
        cagrs[valid] = np.power(final[valid] / initial_value, 1 / n_years) - 1

        # Cash-flow-neutral NAV for drawdown/TWR (unaffected by withdrawals)
        # Use .copy() to prevent numpy mutation of sampled_rets
        log_sampled = np.log1p(sampled_rets.copy())
        nav_paths = np.empty((sampled_rets.shape[0], sampled_rets.shape[1] + 1), dtype=float)
        nav_paths[:, 0] = 1.0
        nav_paths[:, 1:] = np.exp(np.cumsum(log_sampled, axis=1))

        twr_cagrs = np.power(nav_paths[:, -1], 1.0 / n_years) - 1.0
        ann_vol = (np.std(sampled_rets, axis=1, ddof=1) * np.sqrt(12)
                   if sampled_rets.shape[1] > 1
                   else np.zeros(sampled_rets.shape[0], dtype=float))

        if np.all(final == final[0]):
            hist_counts = np.array([len(final)])
            hist_mids = [float(final[0])]
        else:
            hist_counts, hist_edges = np.histogram(
                final, bins=30, range=(max(0, float(np.min(final))), float(np.max(final)))
            )
            hist_mids = ((hist_edges[:-1] + hist_edges[1:]) / 2).tolist()

        stats = {
            "percentiles": pct_lines,
            "time_years": time_years.tolist(),
            "final_median": float(np.median(final)),
            "final_mean": float(np.mean(final)),
            "final_std": float(np.std(final)),
            "final_min": float(np.min(final)),
            "final_max": float(np.max(final)),
            "final_p5": float(np.percentile(final, 5)),
            "final_p25": float(np.percentile(final, 25)),
            "final_p75": float(np.percentile(final, 75)),
            "final_p95": float(np.percentile(final, 95)),
            "sample_trajectories": trajectories[:15, :].tolist(),
            "has_withdrawals": has_withdrawals,
            "histogram": {"counts": hist_counts.tolist(), "mids": hist_mids},
            "min_path_value": float(np.min(trajectories)),
            "median_twr_cagr": float(np.median(twr_cagrs)),
            "median_ann_vol": float(np.median(ann_vol)),
        }

        # Drawdown on cash-flow-neutral NAV (not contaminated by withdrawals)
        peaks = np.maximum.accumulate(nav_paths, axis=1)
        drawdowns = nav_paths / peaks - 1.0
        max_dd = drawdowns.min(axis=1)
        stats["median_max_drawdown"] = float(np.median(max_dd))
        stats["p10_max_drawdown"] = float(np.percentile(max_dd, 10))

        # Benchmark comparison (path-matched)
        if bench_trajectories is not None:
            bench_final = bench_trajectories[:, -1]
            stats["prob_beat_benchmark"] = float(np.mean(final > bench_final) * 100)
            stats["benchmark_final_median"] = float(np.median(bench_final))
            if sampled_bench is not None and sampled_rets.shape[1] > 1:
                excess = sampled_rets - sampled_bench
                excess_mean = np.mean(excess, axis=1)
                excess_vol = np.std(excess, axis=1, ddof=1)
                sharpe = np.full(excess_mean.shape, np.nan, dtype=float)
                valid_sharpe = excess_vol > 0
                sharpe[valid_sharpe] = (
                    excess_mean[valid_sharpe] / excess_vol[valid_sharpe] * np.sqrt(12)
                )
                finite_sharpe = sharpe[np.isfinite(sharpe)]
                stats["median_sharpe_vs_benchmark"] = (
                    float(np.median(finite_sharpe)) if finite_sharpe.size else None
                )
            else:
                stats["median_sharpe_vs_benchmark"] = None
        else:
            stats["prob_beat_benchmark"] = None
            stats["benchmark_final_median"] = None
            stats["median_sharpe_vs_benchmark"] = None

        if has_withdrawals:
            stats["prob_ruin"] = float(np.mean(final <= 0) * 100)
            stats["median_cagr"] = None
            stats["prob_loss"] = None
            ruined = final <= 0
            if np.any(ruined):
                ruin_mask = trajectories[ruined] <= 0
                first_ruin = np.argmax(ruin_mask, axis=1)
                stats["median_ruin_year"] = float(np.median(first_ruin) / 12)
            else:
                stats["median_ruin_year"] = None
        else:
            stats["prob_loss"] = float(np.mean(final < initial_value) * 100)
            stats["prob_ruin"] = None
            stats["median_cagr"] = float(np.median(cagrs))
            stats["median_ruin_year"] = None

        return stats

    nominal_stats = compute_stats(traj_nominal, sampled_returns,
                                  benchmark_nominal, sampled_benchmark)
    real_stats = compute_stats(traj_real, sampled_real_returns,
                               benchmark_real, sampled_benchmark_real)

    return {
        "params": {
            "initial_value": initial_value,
            "n_years": n_years,
            "n_trajectories": n_trajectories,
            "withdrawal_annual": withdrawal_annual,
            "n_historical_months": n_hist,
            "benchmark_name": benchmark_name,
        },
        "nominal": nominal_stats,
        "real": real_stats,
    }

"""Monte Carlo simulation engine for Brazilian portfolio simulator.

All returns are in BRL. Supports nominal and real (IPCA-adjusted) simulations
with optional periodic withdrawals.
"""
import json
import numpy as np
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "historical_returns.json"

TARGET_MONTHS = 240

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


def get_available_indices(data):
    target_start = data["metadata"]["target_start"]
    target_end = data["metadata"]["target_end"]
    target_months_set = {m for m in data["ipca"] if target_start <= m <= target_end}
    target_months = len(target_months_set)
    indices = []
    for key, info in data["indices"].items():
        # Count months this index actually has within the target window
        months_in_window = sum(1 for m in info["returns"] if m in target_months_set)
        indices.append({
            "key": key,
            "name": info["name"],
            "desc": info["desc"],
            "category": info["category"],
            "category_name": CATEGORIES.get(info["category"], info["category"]),
            "months_available": info["months_available"],
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
        cat = idx_info["category"]
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
            f"{idx_info['name']} ({weight*100:.0f}%) — disponível desde "
            f"{idx_info['start_date']}. Nos {total_missing} meses anteriores: "
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
                    n_trajectories=10000, withdrawal_annual=0.0, seed=None):
    """
    Run Monte Carlo simulation using bootstrap resampling.

    Withdrawals are applied monthly (1/12 of annual amount per month),
    adjusted for cumulative inflation in that trajectory.
    """
    if n_years <= 0:
        raise ValueError("Horizonte deve ser > 0")

    rng = np.random.default_rng(seed)

    n_months = n_years * 12
    n_hist = len(portfolio_returns)
    has_withdrawals = withdrawal_annual > 0
    monthly_withdrawal_real = withdrawal_annual / 12.0

    # Bootstrap: sample month indices with replacement
    sampled_idx = rng.integers(0, n_hist, size=(n_trajectories, n_months))
    sampled_returns = portfolio_returns[sampled_idx]
    sampled_ipca = ipca[sampled_idx]

    # Nominal trajectories
    traj_nominal = np.full((n_trajectories, n_months + 1), initial_value, dtype=float)
    cum_inflation = np.ones((n_trajectories, n_months + 1), dtype=float)

    for t in range(n_months):
        # Apply return first
        traj_nominal[:, t + 1] = traj_nominal[:, t] * (1 + sampled_returns[:, t])
        cum_inflation[:, t + 1] = cum_inflation[:, t] * (1 + sampled_ipca[:, t])

        # Monthly withdrawal adjusted for cumulative inflation
        if has_withdrawals:
            withdrawal = monthly_withdrawal_real * cum_inflation[:, t + 1]
            traj_nominal[:, t + 1] = np.maximum(traj_nominal[:, t + 1] - withdrawal, 0.0)

    # Real (inflation-adjusted) trajectories
    # Use explicit copy to prevent numpy in-place optimization on some platforms
    traj_real = traj_nominal.copy()
    np.true_divide(traj_nominal, cum_inflation, out=traj_real)

    # Compute statistics
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    time_years = np.arange(n_months + 1) / 12.0

    def compute_stats(trajectories):
        final = trajectories[:, -1]
        pct_values = np.percentile(trajectories, percentiles, axis=0)
        pct_lines = {p: pct_values[i].tolist() for i, p in enumerate(percentiles)}

        # Compute per-path CAGR then take median
        valid = final > 0
        cagrs = np.full(final.shape, -1.0, dtype=float)
        cagrs[valid] = np.power(final[valid] / initial_value, 1 / n_years) - 1

        # Histogram of final values (30 bins)
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
            "sample_trajectories": trajectories[:8, :].tolist(),
            "has_withdrawals": has_withdrawals,
            "histogram": {"counts": hist_counts.tolist(), "mids": hist_mids},
        }

        # Drawdown metrics
        peaks = np.maximum.accumulate(trajectories, axis=1)
        safe_peaks = np.where(peaks == 0, 1, peaks)
        drawdowns = trajectories / safe_peaks - 1
        max_dd = drawdowns.min(axis=1)
        stats["median_max_drawdown"] = float(np.median(max_dd))
        stats["p10_max_drawdown"] = float(np.percentile(max_dd, 10))

        if has_withdrawals:
            stats["prob_ruin"] = float(np.mean(final <= 0) * 100)
            stats["median_cagr"] = None
            stats["prob_loss"] = None
            # Survival analysis: find first ruin month per trajectory
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

    nominal_stats = compute_stats(traj_nominal)
    real_stats = compute_stats(traj_real)

    return {
        "params": {
            "initial_value": initial_value,
            "n_years": n_years,
            "n_trajectories": n_trajectories,
            "withdrawal_annual": withdrawal_annual,
            "n_historical_months": n_hist,
        },
        "nominal": nominal_stats,
        "real": real_stats,
    }

"""Monte Carlo simulation engine for Brazilian portfolio simulator.

All returns are in BRL. Supports nominal and real (IPCA-adjusted) simulations
with optional periodic withdrawals.
"""
import json
import numpy as np
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "historical_returns.json"

# Categories for the substitution rule
CATEGORIES = {
    "br_fixed_income": "Renda Fixa Brasil",
    "br_equities": "Ações Brasil",
    "intl_equities": "Ações Internacionais",
    "intl_fixed_income": "Renda Fixa Internacional",
    "intl_reits": "REITs Internacionais",
}


def load_data():
    """Load the historical returns dataset."""
    with open(DATA_FILE) as f:
        return json.load(f)


def get_available_indices(data):
    """Return list of available indices with metadata."""
    indices = []
    for key, info in data["indices"].items():
        indices.append({
            "key": key,
            "name": info["name"],
            "desc": info["desc"],
            "category": info["category"],
            "category_name": CATEGORIES.get(info["category"], info["category"]),
            "months_available": info["months_available"],
            "start_date": info["start_date"],
            "end_date": info["end_date"],
            "has_full_history": info["months_available"] >= 236,
        })
    return indices


def build_portfolio_returns(data, allocations):
    """
    Build the portfolio monthly return series applying the substitution rule.

    Args:
        data: The full dataset dict
        allocations: Dict of {index_key: weight} where weights sum to 1.0

    Returns:
        dict with:
            - months: list of YYYY-MM strings
            - portfolio_returns: numpy array of monthly returns
            - ipca: numpy array of monthly IPCA rates (aligned)
            - warnings: list of substitution warning strings
    """
    # Get the full 240-month date range
    all_months = sorted(data["ipca"].keys())
    target_start = data["metadata"]["target_start"]
    target_end = data["metadata"]["target_end"]
    all_months = [m for m in all_months if target_start <= m <= target_end]

    # We need returns for all these months. Some indices may not have data for
    # early months. The IPCA series might be slightly shorter too.
    # Use the intersection of all available months from indices that have allocations
    # and extend with substitution for those that don't cover all months.

    # Group allocations by category
    cat_allocs = {}
    for idx_key, weight in allocations.items():
        if weight <= 0:
            continue
        cat = data["indices"][idx_key]["category"]
        if cat not in cat_allocs:
            cat_allocs[cat] = {}
        cat_allocs[cat][idx_key] = weight

    warnings = []
    # For each month, compute the effective allocation (with substitution)
    # and the weighted portfolio return
    portfolio_returns = []
    months_used = []

    for month in all_months:
        month_return = 0.0
        total_weight_accounted = 0.0

        for cat, idx_weights in cat_allocs.items():
            # Which indices in this category have data for this month?
            available = {}
            unavailable = {}
            for idx_key, weight in idx_weights.items():
                returns = data["indices"][idx_key]["returns"]
                if month in returns:
                    available[idx_key] = weight
                else:
                    unavailable[idx_key] = weight

            if unavailable:
                # Redistribute unavailable weight pro-rata among available indices
                # in the same category
                total_available_weight = sum(available.values())
                total_unavailable_weight = sum(unavailable.values())

                if total_available_weight > 0:
                    for idx_key in available:
                        # Original weight + share of redistributed weight
                        extra = (available[idx_key] / total_available_weight) * total_unavailable_weight
                        available[idx_key] += extra
                else:
                    # No index in this category has data for this month — skip
                    # This weight is lost for this month (edge case)
                    pass

            for idx_key, eff_weight in available.items():
                ret = data["indices"][idx_key]["returns"].get(month, 0.0)
                month_return += eff_weight * ret
                total_weight_accounted += eff_weight

        portfolio_returns.append(month_return)
        months_used.append(month)

    # Generate substitution warnings
    for cat, idx_weights in cat_allocs.items():
        for idx_key, weight in idx_weights.items():
            idx_info = data["indices"][idx_key]
            if idx_info["months_available"] < len(all_months):
                missing = len(all_months) - idx_info["months_available"]
                cat_peers = [k for k in idx_weights if k != idx_key and
                             data["indices"][k]["months_available"] >= len(all_months)]
                if cat_peers:
                    peer_names = ", ".join(data["indices"][k]["name"] for k in cat_peers)
                    warnings.append(
                        f"{idx_info['name']} não possui histórico completo de 20 anos "
                        f"(disponível desde {idx_info['start_date']}). "
                        f"Nos {missing} meses anteriores, a alocação foi redistribuída "
                        f"proporcionalmente para: {peer_names}."
                    )
                else:
                    # Check if any OTHER index in the same category (not in user's allocation)
                    # has full history
                    all_cat_indices = [k for k, v in data["indices"].items()
                                       if v["category"] == cat and k != idx_key]
                    has_full = [k for k in all_cat_indices
                                if data["indices"][k]["months_available"] >= len(all_months)]
                    if has_full:
                        warnings.append(
                            f"{idx_info['name']} não possui histórico de 20 anos "
                            f"(desde {idx_info['start_date']}). Alocação redistribuída "
                            f"para outros índices da mesma categoria nos meses sem dados."
                        )
                    else:
                        warnings.append(
                            f"⚠ {idx_info['name']} não possui 20 anos de histórico e não há "
                            f"substituto na categoria {CATEGORIES.get(cat, cat)}. "
                            f"Simulação usa apenas {idx_info['months_available']} meses."
                        )

    # Align IPCA
    ipca_values = []
    for m in months_used:
        ipca_values.append(data["ipca"].get(m, 0.0))

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

    Args:
        portfolio_returns: numpy array of monthly returns
        ipca: numpy array of monthly IPCA rates (aligned with portfolio_returns)
        initial_value: Starting portfolio value in BRL
        n_years: Simulation horizon in years
        n_trajectories: Number of simulation paths
        withdrawal_annual: Annual withdrawal in BRL (0 = no withdrawals)
        seed: Random seed for reproducibility

    Returns:
        dict with simulation results
    """
    if seed is not None:
        np.random.seed(seed)

    n_months = n_years * 12
    n_hist = len(portfolio_returns)

    # Bootstrap: sample month indices with replacement
    sampled_idx = np.random.randint(0, n_hist, size=(n_trajectories, n_months))
    sampled_returns = portfolio_returns[sampled_idx]
    sampled_ipca = ipca[sampled_idx]

    # Nominal trajectories
    traj_nominal = np.full((n_trajectories, n_months + 1), initial_value, dtype=float)
    cum_inflation = np.ones((n_trajectories, n_months + 1), dtype=float)

    for t in range(n_months):
        # Apply withdrawal at start of each year
        if withdrawal_annual > 0 and t % 12 == 0:
            withdrawal = withdrawal_annual * cum_inflation[:, t]
            traj_nominal[:, t] = np.maximum(traj_nominal[:, t] - withdrawal, 0.0)

        traj_nominal[:, t + 1] = traj_nominal[:, t] * (1 + sampled_returns[:, t])
        cum_inflation[:, t + 1] = cum_inflation[:, t] * (1 + sampled_ipca[:, t])

    # Real (inflation-adjusted) trajectories
    traj_real = traj_nominal / cum_inflation

    # Compute statistics
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    time_years = np.arange(n_months + 1) / 12.0

    def compute_stats(trajectories):
        final = trajectories[:, -1]
        pct_lines = {}
        for p in percentiles:
            pct_lines[p] = np.percentile(trajectories, p, axis=0).tolist()
        return {
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
            "prob_loss": float(np.mean(final < initial_value) * 100),
            "prob_negative": float(np.mean(final <= 0) * 100),
            "median_cagr": float((np.median(final) / initial_value) ** (1 / n_years) - 1) if np.median(final) > 0 else -1.0,
            "sample_trajectories": trajectories[:20, :].tolist(),
        }

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

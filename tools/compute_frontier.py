"""Compute efficient frontier and add to historical_returns.json.

Run after any change to historical data:
    python tools/compute_frontier.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "historical_returns.json"


def compute_data_checksum(data):
    """SHA-256 over sorted index returns + IPCA — deterministic."""
    h = hashlib.sha256()
    # IPCA
    for m in sorted(data["ipca"]):
        h.update(f"ipca:{m}:{data['ipca'][m]:.10f}".encode())
    # Index returns
    for key in sorted(data["indices"]):
        for m in sorted(data["indices"][key]["returns"]):
            h.update(f"{key}:{m}:{data['indices'][key]['returns'][m]:.10f}".encode())
    return "sha256:" + h.hexdigest()


def compute_frontier(data, n_points=20):
    """Compute efficient frontier (real + nominal) from historical data."""
    target_start = data["metadata"]["target_start"]
    target_end = data["metadata"]["target_end"]
    months = sorted(m for m in data["ipca"] if target_start <= m <= target_end)
    n_months = len(months)

    index_keys = sorted(data["indices"].keys())
    n_idx = len(index_keys)
    cdi_idx = index_keys.index("CDI")

    ipca_arr = np.array([data["ipca"][m] for m in months])

    # Build returns matrices
    nominal_matrix = np.empty((n_months, n_idx))
    real_matrix = np.empty((n_months, n_idx))
    for j, key in enumerate(index_keys):
        rets = data["indices"][key]["returns"]
        for i, m in enumerate(months):
            nom = rets.get(m, 0.0)
            nominal_matrix[i, j] = nom
            real_matrix[i, j] = (1.0 + nom) / (1.0 + ipca_arr[i]) - 1.0

    def _compute_one_frontier(returns, label):
        mu = np.mean(returns, axis=0)
        Sigma = np.cov(returns, rowvar=False)
        cdi_mu = mu[cdi_idx]

        max_idx_j = np.argmax(mu)
        max_ret = mu[max_idx_j] * 12
        min_ret = cdi_mu * 12

        n = n_idx
        bounds = [(0, 1) for _ in range(n)]
        targets = np.linspace(min_ret, max_ret, n_points)
        portfolios = []

        for target_ann in targets:
            target_monthly = target_ann / 12.0
            cons = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "ineq", "fun": lambda w, t=target_monthly: w.dot(mu) - t},
            ]

            best_w = None
            best_var = np.inf

            starts = [np.ones(n) / n]
            w_cdi = np.zeros(n); w_cdi[cdi_idx] = 1.0; starts.append(w_cdi)
            w_max = np.zeros(n); w_max[max_idx_j] = 1.0; starts.append(w_max)
            frac = (target_ann - min_ret) / (max_ret - min_ret) if max_ret > min_ret else 0
            w_mix = np.zeros(n); w_mix[cdi_idx] = 1 - frac; w_mix[max_idx_j] = frac
            starts.append(w_mix)

            for w0 in starts:
                try:
                    res = minimize(lambda w: w.dot(Sigma).dot(w), w0, method="SLSQP",
                                  bounds=bounds, constraints=cons,
                                  options={"ftol": 1e-14, "maxiter": 500})
                    if (res.success or res.fun < 1e10) and res.fun < best_var:
                        best_var = res.fun
                        best_w = res.x.copy()
                except Exception:
                    pass

            if best_w is not None:
                w = best_w
                port_mu_ann = w.dot(mu) * 12
                port_std_ann = np.sqrt(w.dot(Sigma).dot(w) * 12)
                sharpe = (port_mu_ann - min_ret) / port_std_ann if port_std_ann > 1e-8 else 0

                # CAGR from real returns (always)
                port_real_monthly = real_matrix.dot(w)
                cum = np.prod(1.0 + port_real_monthly)
                cagr = cum ** (12.0 / n_months) - 1.0

                allocs = {index_keys[j]: round(float(w[j]), 4)
                          for j in range(n) if w[j] > 0.005}
                # Normalize displayed allocations
                alloc_total = sum(allocs.values())
                allocs = {k: round(v / alloc_total, 4) for k, v in allocs.items()}

                portfolios.append({
                    "cagr": round(cagr, 6),
                    "vol": round(port_std_ann, 6),
                    "sharpe": round(sharpe, 4),
                    "allocations": allocs,
                })

        return portfolios

    print("Computing real frontier...")
    real_frontier = _compute_one_frontier(real_matrix, "real")
    print(f"  {len(real_frontier)} points")

    print("Computing nominal frontier...")
    nominal_frontier = _compute_one_frontier(nominal_matrix, "nominal")
    print(f"  {len(nominal_frontier)} points")

    checksum = compute_data_checksum(data)
    print(f"Data checksum: {checksum}")

    return {
        "data_checksum": checksum,
        "real": real_frontier,
        "nominal": nominal_frontier,
    }


def main():
    print(f"Loading {DATA_FILE}...")
    with open(DATA_FILE) as f:
        data = json.load(f)

    frontier = compute_frontier(data)
    data["efficient_frontier"] = frontier

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=None, separators=(",", ":"))

    size_mb = DATA_FILE.stat().st_size / 1024 / 1024
    print(f"\nWrote {DATA_FILE} ({size_mb:.2f} MB)")
    print(f"Checksum: {frontier['data_checksum']}")
    print("\nReal frontier summary:")
    for i, p in enumerate(frontier["real"]):
        alloc_str = ", ".join(f"{k} {v*100:.0f}%" for k, v in
                              sorted(p["allocations"].items(), key=lambda x: -x[1])[:4])
        print(f"  {i+1:>2}. CAGR={p['cagr']*100:.2f}% Vol={p['vol']*100:.2f}% Sharpe={p['sharpe']:.3f}  {alloc_str}")


if __name__ == "__main__":
    main()

"""Compute efficient frontier and add to historical_returns.json.

Run after any change to historical data:
    python tools/compute_frontier.py

Computes frontiers on three periods:
  - full: entire historical window
  - h1:   first half
  - h2:   second half

Each period uses its OWN return data for CAGR, Sharpe, and target range.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "historical_returns.json"
N_POINTS = 100


def compute_data_checksum(data):
    """SHA-256 over sorted index returns + IPCA — deterministic."""
    h = hashlib.sha256()
    for m in sorted(data["ipca"]):
        h.update(f"ipca:{m}:{data['ipca'][m]:.10f}".encode())
    for key in sorted(data["indices"]):
        for m in sorted(data["indices"][key]["returns"]):
            h.update(f"{key}:{m}:{data['indices'][key]['returns'][m]:.10f}".encode())
    return "sha256:" + h.hexdigest()


def _build_matrices(data, month_list):
    """Build nominal and real return matrices for a given month list."""
    index_keys = sorted(data["indices"].keys())
    n_months = len(month_list)
    n_idx = len(index_keys)
    ipca_arr = np.array([data["ipca"][m] for m in month_list])

    nominal = np.empty((n_months, n_idx))
    real = np.empty((n_months, n_idx))
    for j, key in enumerate(index_keys):
        rets = data["indices"][key]["returns"]
        for i, m in enumerate(month_list):
            nom = rets.get(m, 0.0)
            nominal[i, j] = nom
            real[i, j] = (1.0 + nom) / (1.0 + ipca_arr[i]) - 1.0
    return nominal, real


def _optimize_frontier(opt_returns, cagr_returns, n_months_cagr, cdi_idx,
                       index_keys, n_points):
    """Compute min-variance portfolios.

    opt_returns:  matrix to optimize on (mean/cov computed from this)
    cagr_returns: matrix to compute CAGR from (same period as opt_returns)
    n_months_cagr: number of months in cagr_returns
    """
    mu = np.mean(opt_returns, axis=0)
    Sigma = np.cov(opt_returns, rowvar=False)
    cdi_mu_ann = mu[cdi_idx] * 12

    # Each period uses its own target range
    min_ret = mu[cdi_idx] * 12
    max_ret = np.max(mu) * 12

    n = len(index_keys)
    bounds = [(0, 1)] * n
    targets_monthly = np.linspace(min_ret / 12, max_ret / 12, n_points)

    portfolios = []
    prev_w = None

    for target_monthly in targets_monthly:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, t=target_monthly: w.dot(mu) - t},
        ]

        best_w = None
        best_var = np.inf

        starts = [np.ones(n) / n]
        w_cdi = np.zeros(n); w_cdi[cdi_idx] = 1.0; starts.append(w_cdi)
        max_j = np.argmax(mu)
        w_max = np.zeros(n); w_max[max_j] = 1.0; starts.append(w_max)
        if max_ret > min_ret:
            frac = max(0, min(1, (target_monthly * 12 - min_ret) / (max_ret - min_ret)))
            w_mix = np.zeros(n); w_mix[cdi_idx] = 1 - frac; w_mix[max_j] = frac
            starts.append(w_mix)
        if prev_w is not None:
            starts.append(prev_w)

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
            prev_w = w.copy()
            port_mu_ann = w.dot(mu) * 12
            port_std_ann = np.sqrt(w.dot(Sigma).dot(w) * 12)
            sharpe = (port_mu_ann - cdi_mu_ann) / port_std_ann if port_std_ann > 1e-8 else 0

            # CAGR from this period's returns
            port_monthly = cagr_returns.dot(w)
            cum = np.prod(1.0 + port_monthly)
            cagr = cum ** (12.0 / n_months_cagr) - 1.0

            allocs = {index_keys[j]: round(float(w[j]), 4)
                      for j in range(n) if w[j] > 0.005}
            alloc_total = sum(allocs.values())
            allocs = {k: round(v / alloc_total, 4) for k, v in allocs.items()}

            portfolios.append({
                "cagr": round(cagr, 6),
                "vol": round(port_std_ann, 6),
                "sharpe": round(sharpe, 4),
                "allocations": allocs,
            })

    return portfolios


def compute_frontier(data, n_points=N_POINTS):
    """Compute efficient frontier on full, first-half, and second-half periods."""
    target_start = data["metadata"]["target_start"]
    target_end = data["metadata"]["target_end"]
    months = sorted(m for m in data["ipca"] if target_start <= m <= target_end)
    n_months = len(months)
    mid = n_months // 2

    months_h1 = months[:mid]
    months_h2 = months[mid:]
    print(f"  Full: {months[0]}..{months[-1]} ({n_months} months)")
    print(f"  H1:   {months_h1[0]}..{months_h1[-1]} ({len(months_h1)} months)")
    print(f"  H2:   {months_h2[0]}..{months_h2[-1]} ({len(months_h2)} months)")

    index_keys = sorted(data["indices"].keys())
    cdi_idx = index_keys.index("CDI")

    nom_full, real_full = _build_matrices(data, months)
    nom_h1, real_h1 = _build_matrices(data, months_h1)
    nom_h2, real_h2 = _build_matrices(data, months_h2)

    def _compute_set(ret_full, ret_h1, ret_h2, label):
        """Compute frontiers where each period uses its own data for everything."""
        print(f"  {label} full...")
        full = _optimize_frontier(ret_full, ret_full, len(months),
                                  cdi_idx, index_keys, n_points)
        print(f"    {len(full)} points")

        print(f"  {label} H1...")
        h1 = _optimize_frontier(ret_h1, ret_h1, len(months_h1),
                                cdi_idx, index_keys, n_points)
        print(f"    {len(h1)} points")

        print(f"  {label} H2...")
        h2 = _optimize_frontier(ret_h2, ret_h2, len(months_h2),
                                cdi_idx, index_keys, n_points)
        print(f"    {len(h2)} points")

        return {"full": full, "h1": h1, "h2": h2}

    print("Computing real frontiers...")
    real_set = _compute_set(real_full, real_h1, real_h2, "real")

    print("Computing nominal frontiers...")
    nom_set = _compute_set(nom_full, nom_h1, nom_h2, "nominal")

    checksum = compute_data_checksum(data)
    print(f"Data checksum: {checksum}")

    return {
        "data_checksum": checksum,
        "periods": {
            "h1": f"{months_h1[0]} a {months_h1[-1]}",
            "h2": f"{months_h2[0]} a {months_h2[-1]}",
        },
        "real": real_set,
        "nominal": nom_set,
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
    print(f"\nReal frontier (full, every 10th):")
    for i in range(0, len(frontier["real"]["full"]), 10):
        p = frontier["real"]["full"][i]
        alloc_str = ", ".join(f"{k} {v*100:.0f}%" for k, v in
                              sorted(p["allocations"].items(), key=lambda x: -x[1])[:4])
        print(f"  {i+1:>3}. CAGR={p['cagr']*100:.2f}% Sharpe={p['sharpe']:.3f}  {alloc_str}")


if __name__ == "__main__":
    main()

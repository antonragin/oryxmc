"""OryxMC - Monte Carlo Portfolio Simulator for Brazilian Investors."""
import os
import math
import hmac
import secrets
from urllib.parse import urlsplit
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
from functools import wraps
import engine

ALLOCATION_TOLERANCE = 0.006  # ±0.6% — shared across frontend and backend
MAX_SIM_BYTES = 160_000_000  # OOM protection for 512MB / 2-worker deploy
SIM_YEARS = (5, 10, 15, 20, 25, 30, 40, 50)
SIM_TRAJECTORIES = (1000, 5000, 10000, 20000)


def _estimate_sim_bytes(n_years, n_trajectories):
    """Conservative upper bound for peak memory of run_monte_carlo()."""
    n_months = n_years * 12
    month_cells = n_trajectories * n_months
    path_cells = n_trajectories * (n_months + 1)
    # Peak array count: 9 month-sized (sampled_idx as int64, sampled_returns,
    # sampled_ipca, log_ret, log_ipca, sampled_real_returns, cum_log_ipca,
    # sampled_benchmark, sampled_benchmark_real) + 5 path-sized (cum_inflation,
    # traj_nominal, traj_real, benchmark_nominal, benchmark_real).
    # compute_stats nav_paths/peaks are temporary within each call.
    return (9 * month_cells + 5 * path_cells) * 8

IS_PROD = bool(os.environ.get("RENDER") or os.environ.get("FLASK_ENV") == "production")

# Precompute valid year/trajectory combinations once
ALLOWED_RUNS = {
    y: sorted(t for t in SIM_TRAJECTORIES if _estimate_sim_bytes(y, t) <= MAX_SIM_BYTES)
    for y in SIM_YEARS
}


def _parse_float(params, name, default=None, allow_zero=True):
    """Parse a float field from request params, rejecting bools and non-finites."""
    raw = params.get(name, default)
    if isinstance(raw, bool):
        raise ValueError(f"{name} inválido")
    val = float(raw)
    if not math.isfinite(val):
        raise ValueError(f"{name} inválido")
    if not allow_zero and val <= 0:
        raise ValueError(f"{name} deve ser > 0")
    return val


def _parse_int(params, name, default=None):
    """Parse an integer field from request params, accepting int-like floats and strings."""
    raw = params.get(name, default)
    if isinstance(raw, bool):
        raise ValueError(f"{name} inválido")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        # Try direct int parse first (preserves precision for large integers)
        try:
            return int(raw)
        except ValueError:
            pass
    val = float(raw)
    if val != int(val):
        raise ValueError(f"{name} deve ser inteiro")
    return int(val)


app = Flask(__name__)
# Trust X-Forwarded-For and X-Forwarded-Proto from Render's reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
_secret = os.environ.get("SECRET_KEY")
APP_PASSWORD = os.environ.get("APP_PASSWORD")
if IS_PROD and not _secret:
    raise RuntimeError("SECRET_KEY must be set in production")
if IS_PROD and not APP_PASSWORD:
    raise RuntimeError("APP_PASSWORD must be set in production")
app.secret_key = _secret or secrets.token_hex(32)
APP_PASSWORD = APP_PASSWORD or "oryx2026"
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB max request
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = IS_PROD

# Load data once at startup
DATA = engine.load_data()
engine.validate_data(DATA)
INDICES = engine.get_available_indices(DATA)
BENCHMARK_CDI = engine.build_portfolio_returns(DATA, {"CDI": 1.0})


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Não autenticado"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.errorhandler(RequestEntityTooLarge)
def handle_413(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Requisição muito grande"}), 413
    return "Requisição muito grande", 413


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("Cache-Control", "no-store, no-cache, must-revalidate, private")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    if IS_PROD:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.plot.ly; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )
    return response


@app.before_request
def check_origin():
    """Reject cross-origin state-changing requests (CSRF defense-in-depth)."""
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        # Enforce JSON content type on API routes (immune to <form> CSRF)
        if request.path.startswith("/api/") and not request.is_json:
            return jsonify({"error": "Content-Type deve ser application/json"}), 415

        origin = request.headers.get("Origin")
        if not origin:
            # Fail closed if Origin is omitted on API routes
            if request.path.startswith("/api/"):
                return jsonify({"error": "Origem ausente"}), 403
        else:
            try:
                origin_host = urlsplit(origin).netloc
            except ValueError:
                origin_host = None
            if origin_host != request.host:
                if request.path.startswith("/api/"):
                    return jsonify({"error": "Origem inválida"}), 403
                return "Origem inválida", 403


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        pwd = request.form.get("password", "")
        if hmac.compare_digest(pwd, APP_PASSWORD):
            session.clear()
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Senha incorreta"
    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/api/indices")
@login_required
def api_indices():
    return jsonify({
        "indices": INDICES,
        "metadata": {
            "target_start": DATA["metadata"]["target_start"],
            "target_end": DATA["metadata"]["target_end"],
            "n_months": len([m for m in DATA["ipca"]
                            if DATA["metadata"]["target_start"] <= m <= DATA["metadata"]["target_end"]]),
            "allocation_tolerance_pct": ALLOCATION_TOLERANCE * 100,
            "allowed_runs": {str(y): v for y, v in ALLOWED_RUNS.items()},
            "year_options": sorted(y for y, runs in ALLOWED_RUNS.items() if runs),
        },
    })


@app.route("/api/simulate", methods=["POST"])
@login_required
def api_simulate():
    try:
        params = request.get_json(silent=True)
        if not isinstance(params, dict):
            return jsonify({"error": "JSON inválido"}), 400

        allocations = params.get("allocations", {})
        if not isinstance(allocations, dict):
            return jsonify({"error": "allocations inválido"}), 400

        try:
            initial_value = _parse_float(params, "initial_value", default=1000000)
            n_years = _parse_int(params, "n_years", default=10)
            withdrawal_annual = _parse_float(params, "withdrawal_annual", default=0)
            n_trajectories = _parse_int(params, "n_trajectories", default=10000)
        except (TypeError, ValueError, OverflowError):
            return jsonify({"error": "Parâmetros numéricos inválidos"}), 400

        # Validate parameters
        if initial_value <= 0 or initial_value > 1e12:
            return jsonify({"error": "Valor inicial inválido"}), 400
        if n_years <= 0 or n_years > 50:
            return jsonify({"error": "Horizonte deve ser entre 1 e 50 anos"}), 400

        # Handle withdrawal mode (fixed BRL vs % of initial capital / SWR)
        withdrawal_mode = params.get("withdrawal_mode", "fixed")
        if withdrawal_mode not in ("fixed", "percent"):
            return jsonify({"error": "Modo de retirada inválido"}), 400
        if withdrawal_mode == "percent":
            try:
                withdrawal_percent = _parse_float(params, "withdrawal_percent", default=0)
            except (TypeError, ValueError, OverflowError):
                return jsonify({"error": "Taxa SWR inválida"}), 400
            if withdrawal_percent < 0 or withdrawal_percent > 100:
                return jsonify({"error": "Taxa SWR deve ser entre 0% e 100%"}), 400
            withdrawal_annual = initial_value * (withdrawal_percent / 100.0)
        else:
            withdrawal_percent = None
            if withdrawal_annual < 0:
                return jsonify({"error": "Retirada anual não pode ser negativa"}), 400

        if n_trajectories <= 0 or n_trajectories > 20000:
            return jsonify({"error": "Número de trajetórias deve ser entre 1 e 20.000"}), 400

        # Enforce allowed year/trajectory combinations (OOM protection)
        if n_years not in ALLOWED_RUNS or n_trajectories not in ALLOWED_RUNS[n_years]:
            return jsonify({"error": "Combinação de horizonte e trajetórias não permitida neste servidor"}), 400

        # Parse bootstrap mode
        bootstrap_mode = params.get("bootstrap_mode", "iid")
        if bootstrap_mode not in engine.BOOTSTRAP_MODES:
            return jsonify({"error": "Modo de bootstrap inválido"}), 400
        block_size = 12
        if bootstrap_mode == "block":
            try:
                block_size = _parse_int(params, "block_size", default=12)
            except (TypeError, ValueError, OverflowError):
                return jsonify({"error": "Tamanho de bloco inválido"}), 400
            if block_size not in engine.BLOCK_SIZES:
                return jsonify({"error": "Tamanho de bloco deve ser 6 ou 12"}), 400

        # Parse optional seed for reproducibility
        raw_seed = params.get("seed")
        if raw_seed in (None, ""):
            seed = secrets.randbits(53)  # Stay within JS Number.MAX_SAFE_INTEGER
        else:
            try:
                seed = _parse_int(params, "seed")
            except (TypeError, ValueError, OverflowError):
                return jsonify({"error": "Seed inválido"}), 400
            if seed < 0 or seed > 2**53 - 1:
                return jsonify({"error": "Seed inválido"}), 400

        # Validate allocations
        clean_allocations = {}
        for key, val in allocations.items():
            if key not in DATA["indices"]:
                return jsonify({"error": f"Índice desconhecido: {key}"}), 400
            if isinstance(val, bool):
                return jsonify({"error": f"Alocação inválida: {key}"}), 400
            try:
                val = float(val)
            except (TypeError, ValueError, OverflowError):
                return jsonify({"error": f"Alocação inválida: {key}"}), 400
            if not math.isfinite(val) or val < 0:
                return jsonify({"error": f"Alocação inválida: {key}"}), 400
            clean_allocations[key] = val

        allocations = clean_allocations
        total = sum(allocations.values())
        if not math.isfinite(total) or abs(total - 1.0) > ALLOCATION_TOLERANCE + 1e-9:
            return jsonify({"error": f"Alocações devem somar 100% (atual: {total*100:.1f}%)"}), 400

        # Build portfolio returns with substitution
        portfolio = engine.build_portfolio_returns(DATA, allocations)

        # Run simulation
        results = engine.run_monte_carlo(
            portfolio["portfolio_returns"],
            portfolio["ipca"],
            initial_value=initial_value,
            n_years=n_years,
            n_trajectories=n_trajectories,
            withdrawal_annual=withdrawal_annual,
            seed=seed,
            benchmark_returns=BENCHMARK_CDI["portfolio_returns"],
            benchmark_name="CDI",
            bootstrap_mode=bootstrap_mode,
            block_size=block_size,
        )
        results["warnings"] = portfolio["warnings"]
        results["params"]["seed"] = seed

        return jsonify(results)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        error_id = secrets.token_hex(4)
        app.logger.exception("api_simulate failed [error_id=%s]", error_id)
        return jsonify({"error": "Erro interno do servidor", "error_id": error_id}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true")
    app.run(host="0.0.0.0", port=port, debug=debug)

"""OryxMC - Monte Carlo Portfolio Simulator for Brazilian Investors."""
import os
import math
import hmac
import secrets
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
import engine

ALLOCATION_TOLERANCE = 0.006  # ±0.6% — shared across frontend and backend

IS_PROD = bool(os.environ.get("RENDER") or os.environ.get("FLASK_ENV") == "production")

app = Flask(__name__)
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


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("Cache-Control", "no-store, no-cache, must-revalidate, private")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
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


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        pwd = request.form.get("password", "")
        if hmac.compare_digest(pwd, APP_PASSWORD):
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
            iv_raw = params.get("initial_value", 1000000)
            if isinstance(iv_raw, bool):
                return jsonify({"error": "Valor inicial inválido"}), 400
            initial_value = float(iv_raw)
            n_years_raw = params.get("n_years", 10)
            if not isinstance(n_years_raw, int) or isinstance(n_years_raw, bool):
                n_years_f = float(n_years_raw)
                if n_years_f != int(n_years_f):
                    return jsonify({"error": "Horizonte deve ser um número inteiro"}), 400
                n_years_raw = int(n_years_f)
            n_years = n_years_raw
            wa_raw = params.get("withdrawal_annual", 0)
            if isinstance(wa_raw, bool):
                return jsonify({"error": "Retirada anual inválida"}), 400
            withdrawal_annual = float(wa_raw)
            n_traj_raw = params.get("n_trajectories", 10000)
            if not isinstance(n_traj_raw, int) or isinstance(n_traj_raw, bool):
                n_traj_f = float(n_traj_raw)
                if n_traj_f != int(n_traj_f):
                    return jsonify({"error": "Número de trajetórias deve ser inteiro"}), 400
                n_traj_raw = int(n_traj_f)
            n_trajectories = n_traj_raw
        except (TypeError, ValueError, OverflowError):
            return jsonify({"error": "Parâmetros numéricos inválidos"}), 400

        # Validate parameters (reject NaN/Infinity)
        if not math.isfinite(initial_value) or initial_value <= 0 or initial_value > 1e12:
            return jsonify({"error": "Valor inicial inválido"}), 400
        if n_years <= 0 or n_years > 50:
            return jsonify({"error": "Horizonte deve ser entre 1 e 50 anos"}), 400
        # Handle withdrawal mode (fixed BRL vs % of initial capital / SWR)
        withdrawal_mode = params.get("withdrawal_mode", "fixed")
        if withdrawal_mode not in ("fixed", "percent"):
            return jsonify({"error": "Modo de retirada inválido"}), 400
        if withdrawal_mode == "percent":
            wp_raw = params.get("withdrawal_percent", 0)
            if isinstance(wp_raw, bool):
                return jsonify({"error": "Taxa SWR inválida"}), 400
            try:
                withdrawal_percent = float(wp_raw)
            except (TypeError, ValueError, OverflowError):
                return jsonify({"error": "Taxa SWR inválida"}), 400
            if not math.isfinite(withdrawal_percent) or withdrawal_percent < 0 or withdrawal_percent > 100:
                return jsonify({"error": "Taxa SWR deve ser entre 0% e 100%"}), 400
            withdrawal_annual = initial_value * (withdrawal_percent / 100.0)
        else:
            withdrawal_percent = None
            if not math.isfinite(withdrawal_annual) or withdrawal_annual < 0:
                return jsonify({"error": "Retirada anual não pode ser negativa"}), 400
        if n_trajectories <= 0 or n_trajectories > 20000:
            return jsonify({"error": "Número de trajetórias deve ser entre 1 e 20.000"}), 400

        # Reject oversized requests (OOM protection for 512MB / 2-worker deploy)
        # Engine allocates ~6 full arrays of size (n_traj, n_months): ~48 bytes/cell
        n_months_est = n_years * 12
        estimated_bytes = n_trajectories * n_months_est * 48 + n_trajectories * (n_months_est + 1) * 24
        if estimated_bytes > 120_000_000:
            return jsonify({"error": "Simulação muito grande para o servidor atual"}), 400

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
        if not math.isfinite(total) or abs(total - 1.0) > ALLOCATION_TOLERANCE:
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
            benchmark_returns=BENCHMARK_CDI["portfolio_returns"],
            benchmark_name="CDI",
        )
        results["warnings"] = portfolio["warnings"]

        return jsonify(results)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("api_simulate failed")
        return jsonify({"error": "Erro interno do servidor"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true")
    app.run(host="0.0.0.0", port=port, debug=debug)

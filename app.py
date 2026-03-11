"""OryxMC - Monte Carlo Portfolio Simulator for Brazilian Investors."""
import os
import math
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
import engine

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32).hex()
APP_PASSWORD = os.environ.get("APP_PASSWORD", "oryx2026")

# Load data once at startup
DATA = engine.load_data()
INDICES = engine.get_available_indices(DATA)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Não autenticado"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == APP_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Senha incorreta"
    return render_template("login.html", error=error)


@app.route("/logout")
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
    return jsonify(INDICES)


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
            initial_value = float(params.get("initial_value", 1000000))
            n_years = int(params.get("n_years", 10))
            withdrawal_annual = float(params.get("withdrawal_annual", 0))
            n_trajectories = min(int(params.get("n_trajectories", 10000)), 20000)
        except (TypeError, ValueError, OverflowError):
            return jsonify({"error": "Parâmetros numéricos inválidos"}), 400

        # Validate parameters (reject NaN/Infinity)
        if not math.isfinite(initial_value) or initial_value <= 0 or initial_value > 1e12:
            return jsonify({"error": "Valor inicial inválido"}), 400
        if n_years <= 0 or n_years > 50:
            return jsonify({"error": "Horizonte deve ser entre 1 e 50 anos"}), 400
        if not math.isfinite(withdrawal_annual) or withdrawal_annual < 0:
            return jsonify({"error": "Retirada anual não pode ser negativa"}), 400
        if n_trajectories <= 0:
            return jsonify({"error": "Número de trajetórias inválido"}), 400

        # Reject oversized requests (OOM protection)
        if n_trajectories * n_years * 12 > 4_000_000:
            return jsonify({"error": "Combinação de horizonte e trajetórias muito grande"}), 400

        # Validate allocations
        clean_allocations = {}
        for key, val in allocations.items():
            if key not in DATA["indices"]:
                return jsonify({"error": f"Índice desconhecido: {key}"}), 400
            try:
                val = float(val)
            except (TypeError, ValueError, OverflowError):
                return jsonify({"error": f"Alocação inválida: {key}"}), 400
            if not math.isfinite(val) or val < 0:
                return jsonify({"error": f"Alocação inválida: {key}"}), 400
            clean_allocations[key] = val

        allocations = clean_allocations
        total = sum(allocations.values())
        if not math.isfinite(total) or abs(total - 1.0) > 0.011:
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
        )
        results["warnings"] = portfolio["warnings"]

        return jsonify(results)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

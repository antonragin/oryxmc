"""OryxMC - Monte Carlo Portfolio Simulator for Brazilian Investors."""
import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
import engine

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "oryxmc-dev-secret-2026")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "oryx2026")

# Load data once at startup
DATA = engine.load_data()
INDICES = engine.get_available_indices(DATA)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
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
        params = request.get_json()
        allocations = params.get("allocations", {})
        initial_value = float(params.get("initial_value", 1000000))
        n_years = int(params.get("n_years", 10))
        withdrawal_annual = float(params.get("withdrawal_annual", 0))
        n_trajectories = min(int(params.get("n_trajectories", 10000)), 20000)

        # Validate allocations
        total = sum(allocations.values())
        if abs(total - 1.0) > 0.01:
            return jsonify({"error": f"Alocações devem somar 100% (atual: {total*100:.1f}%)"}), 400

        for key in allocations:
            if key not in DATA["indices"]:
                return jsonify({"error": f"Índice desconhecido: {key}"}), 400

        # Build portfolio returns
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

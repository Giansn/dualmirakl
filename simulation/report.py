"""
Self-contained HTML report generator for dualmirakl simulation output.

Produces interactive reports with Plotly.js charts (loaded from CDN),
rendered via Jinja2 templates. Reports include fan charts, spaghetti plots,
tornado diagrams, variance decomposition, and scenario tree visualization.

No local Plotly install required — charts are rendered client-side.

Usage:
    from simulation.report import generate_report

    html = generate_report(
        ensemble_result=result,
        dynamics_analysis=analysis,
        scenario_tree=tree,
    )
    Path("report.html").write_text(html)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from jinja2 import Template

logger = logging.getLogger(__name__)

# ── Plotly JSON chart builders ────────────────────────────────────────────────

def _fan_chart_json(percentile_bands: dict) -> str:
    """Build Plotly JSON for a fan chart (nested confidence bands)."""
    steps = sorted(int(k) for k in percentile_bands.keys())
    bands = percentile_bands

    traces = []

    # 95% band (lightest)
    p5 = [bands[str(s) if str(s) in bands else s]["p5"] for s in steps]
    p95 = [bands[str(s) if str(s) in bands else s]["p95"] for s in steps]
    traces.append({
        "x": steps + steps[::-1],
        "y": p95 + p5[::-1],
        "fill": "toself",
        "fillcolor": "rgba(68, 119, 170, 0.15)",
        "line": {"color": "transparent"},
        "name": "95% CI",
        "showlegend": True,
    })

    # 50% band (darker)
    p25 = [bands[str(s) if str(s) in bands else s]["p25"] for s in steps]
    p75 = [bands[str(s) if str(s) in bands else s]["p75"] for s in steps]
    traces.append({
        "x": steps + steps[::-1],
        "y": p75 + p25[::-1],
        "fill": "toself",
        "fillcolor": "rgba(68, 119, 170, 0.3)",
        "line": {"color": "transparent"},
        "name": "50% CI",
        "showlegend": True,
    })

    # Median line
    median = [bands[str(s) if str(s) in bands else s]["median"] for s in steps]
    traces.append({
        "x": steps,
        "y": median,
        "mode": "lines",
        "line": {"color": "#4477AA", "width": 2},
        "name": "Median",
    })

    layout = {
        "title": "Ensemble Fan Chart",
        "xaxis": {"title": "Tick"},
        "yaxis": {"title": "Mean Score", "range": [0, 1]},
        "margin": {"t": 40, "b": 40, "l": 50, "r": 20},
        "height": 350,
    }
    return json.dumps({"data": traces, "layout": layout})


def _spaghetti_plot_json(all_score_logs: list[list[list[float]]], max_runs: int = 30) -> str:
    """Build Plotly JSON for a spaghetti plot (individual run trajectories)."""
    traces = []
    n_runs = min(len(all_score_logs), max_runs)

    for r in range(n_runs):
        # Mean score across agents per tick
        n_ticks = len(all_score_logs[r][0]) if all_score_logs[r] else 0
        means = []
        for t in range(n_ticks):
            agent_scores = [all_score_logs[r][a][t] for a in range(len(all_score_logs[r]))
                          if t < len(all_score_logs[r][a])]
            means.append(float(np.mean(agent_scores)) if agent_scores else 0.0)

        traces.append({
            "x": list(range(1, n_ticks + 1)),
            "y": means,
            "mode": "lines",
            "line": {"color": "rgba(68, 119, 170, 0.2)", "width": 1},
            "showlegend": False,
            "hoverinfo": "skip",
        })

    layout = {
        "title": f"Individual Run Trajectories ({n_runs} runs)",
        "xaxis": {"title": "Tick"},
        "yaxis": {"title": "Mean Score", "range": [0, 1]},
        "margin": {"t": 40, "b": 40, "l": 50, "r": 20},
        "height": 300,
    }
    return json.dumps({"data": traces, "layout": layout})


def _tornado_chart_json(sobol_indices: dict) -> str:
    """Build Plotly JSON for a tornado diagram (Sobol S1/ST horizontal bars)."""
    if not sobol_indices:
        return json.dumps({"data": [], "layout": {"title": "No sensitivity data"}})

    s1 = sobol_indices.get("S1", {})
    st = sobol_indices.get("ST", {})
    params = sorted(s1.keys(), key=lambda k: st.get(k, s1.get(k, 0)), reverse=True)

    traces = [
        {
            "y": params,
            "x": [round(st.get(p, 0), 4) for p in params],
            "type": "bar",
            "orientation": "h",
            "name": "Total-order (ST)",
            "marker": {"color": "rgba(68, 119, 170, 0.4)"},
        },
        {
            "y": params,
            "x": [round(s1.get(p, 0), 4) for p in params],
            "type": "bar",
            "orientation": "h",
            "name": "First-order (S1)",
            "marker": {"color": "#4477AA"},
        },
    ]

    layout = {
        "title": "Parameter Sensitivity (Sobol Indices)",
        "xaxis": {"title": "Sensitivity Index", "range": [0, 1]},
        "barmode": "overlay",
        "margin": {"t": 40, "b": 40, "l": 120, "r": 20},
        "height": max(200, len(params) * 40 + 80),
    }
    return json.dumps({"data": traces, "layout": layout})


def _variance_pie_json(variance_decomposition: dict) -> str:
    """Build Plotly JSON for variance decomposition pie chart."""
    vd = variance_decomposition
    if not vd or vd.get("var_total", 0) == 0:
        return json.dumps({"data": [], "layout": {"title": "No variance data"}})

    labels = ["Epistemic (parameters)", "Within (aleatory + LLM)"]
    values = [vd.get("var_epistemic", 0), vd.get("var_within", 0)]
    colors = ["#4477AA", "#CC6677"]

    traces = [{
        "labels": labels,
        "values": [round(v, 6) for v in values],
        "type": "pie",
        "marker": {"colors": colors},
        "textinfo": "label+percent",
        "hole": 0.3,
    }]

    layout = {
        "title": "Variance Decomposition",
        "margin": {"t": 40, "b": 20, "l": 20, "r": 20},
        "height": 300,
    }
    return json.dumps({"data": traces, "layout": layout})


def _convergence_plot_json(metric_values: list[float], cv_threshold: float = 0.05) -> str:
    """Build Plotly JSON for convergence monitoring plot."""
    if not metric_values or len(metric_values) < 2:
        return json.dumps({"data": [], "layout": {"title": "No convergence data"}})

    n = len(metric_values)
    cvs = []
    for i in range(2, n + 1):
        arr = np.array(metric_values[:i])
        cv = float(np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else 0.0
        cvs.append(round(cv, 4))

    traces = [
        {
            "x": list(range(2, n + 1)),
            "y": cvs,
            "mode": "lines+markers",
            "name": "CV",
            "line": {"color": "#4477AA"},
        },
        {
            "x": [2, n],
            "y": [cv_threshold, cv_threshold],
            "mode": "lines",
            "name": f"Threshold ({cv_threshold})",
            "line": {"color": "#CC6677", "dash": "dash"},
        },
    ]

    layout = {
        "title": "Ensemble Convergence",
        "xaxis": {"title": "Number of Runs"},
        "yaxis": {"title": "Coefficient of Variation"},
        "margin": {"t": 40, "b": 40, "l": 50, "r": 20},
        "height": 280,
    }
    return json.dumps({"data": traces, "layout": layout})


# ── Jinja2 HTML template ─────────────────────────────────────────────────────

REPORT_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>dualmirakl Simulation Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f8f9fa; color: #2d3436; line-height: 1.6; }
  .container { max-width: 1000px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 1.5rem; margin-bottom: 4px; color: #2d3436; }
  h2 { font-size: 1.1rem; margin: 24px 0 8px; color: #4477AA;
       border-bottom: 2px solid #4477AA; padding-bottom: 4px; }
  .subtitle { color: #636e72; font-size: 0.85rem; margin-bottom: 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: white; border-radius: 8px; padding: 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .card-full { grid-column: 1 / -1; }
  .metric { display: flex; justify-content: space-between; padding: 4px 0;
            border-bottom: 1px solid #f0f0f0; font-size: 0.9rem; }
  .metric-label { color: #636e72; }
  .metric-value { font-weight: 600; font-variant-numeric: tabular-nums; }
  .chart { width: 100%; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }
  th { background: #f1f3f5; font-weight: 600; color: #4477AA; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: 0.75rem; font-weight: 600; }
  .badge-green { background: #d4edda; color: #155724; }
  .badge-yellow { background: #fff3cd; color: #856404; }
  .badge-red { background: #f8d7da; color: #721c24; }
  footer { margin-top: 32px; text-align: center; color: #b2bec3; font-size: 0.75rem; }
</style>
</head>
<body>
<div class="container">

<h1>dualmirakl Simulation Report</h1>
<div class="subtitle">Generated {{ timestamp }} | {{ experiment_id }}</div>

<!-- Summary metrics -->
<h2>Summary</h2>
<div class="grid">
  <div class="card">
    <div class="metric"><span class="metric-label">Runs completed</span>
      <span class="metric-value">{{ n_completed }}</span></div>
    <div class="metric"><span class="metric-label">Runs failed</span>
      <span class="metric-value">{{ n_failed }}</span></div>
    <div class="metric"><span class="metric-label">Convergence</span>
      <span class="metric-value">
        {% if convergence_achieved %}<span class="badge badge-green">Yes (CV={{ final_cv }})</span>
        {% else %}<span class="badge badge-yellow">No (CV={{ final_cv }})</span>{% endif %}
      </span></div>
  </div>
  <div class="card">
    {% if variance_decomposition %}
    <div class="metric"><span class="metric-label">Var epistemic</span>
      <span class="metric-value">{{ pct_epistemic }}%</span></div>
    <div class="metric"><span class="metric-label">Var within (aleatory+LLM)</span>
      <span class="metric-value">{{ pct_within }}%</span></div>
    {% endif %}
    {% for key, val in scoring_metrics.items() %}
    <div class="metric"><span class="metric-label">{{ key }}</span>
      <span class="metric-value">{{ val }}</span></div>
    {% endfor %}
  </div>
</div>

<!-- Fan chart -->
{% if fan_chart_json %}
<h2>Ensemble Fan Chart</h2>
<div class="card card-full">
  <div id="fan-chart" class="chart"></div>
</div>
{% endif %}

<!-- Spaghetti plot -->
{% if spaghetti_json %}
<div class="card card-full">
  <div id="spaghetti-plot" class="chart"></div>
</div>
{% endif %}

<!-- Tornado + Variance -->
<div class="grid">
  {% if tornado_json %}
  <div class="card">
    <div id="tornado-chart" class="chart"></div>
  </div>
  {% endif %}
  {% if variance_pie_json %}
  <div class="card">
    <div id="variance-pie" class="chart"></div>
  </div>
  {% endif %}
</div>

<!-- Convergence -->
{% if convergence_json %}
<h2>Convergence</h2>
<div class="card card-full">
  <div id="convergence-plot" class="chart"></div>
</div>
{% endif %}

<!-- Scenario tree -->
{% if scenario_tree %}
<h2>Scenario Tree</h2>
<div class="card card-full">
  <table>
    <tr><th>Branch</th><th>Probability</th><th>Mean Score</th><th>Std</th><th>Runs</th></tr>
    {% for branch in scenario_branches %}
    <tr>
      <td>{{ branch.id }}</td>
      <td>{{ branch.probability }}%</td>
      <td>{{ branch.mean }}</td>
      <td>{{ branch.std }}</td>
      <td>{{ branch.n_runs }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

<!-- Calibration -->
{% if calibration %}
<h2>Calibration (ABC-SMC)</h2>
<div class="card card-full">
  <table>
    <tr><th>Parameter</th><th>Posterior Mean</th><th>Posterior Std</th></tr>
    {% for name, mean, std in calibration_params %}
    <tr><td>{{ name }}</td><td>{{ mean }}</td><td>{{ std }}</td></tr>
    {% endfor %}
  </table>
  <div class="metric" style="margin-top:8px">
    <span class="metric-label">Populations</span>
    <span class="metric-value">{{ calibration.n_populations }}</span>
  </div>
  <div class="metric">
    <span class="metric-label">Acceptance rate</span>
    <span class="metric-value">{{ calibration.acceptance_rate }}</span>
  </div>
</div>
{% endif %}

<footer>dualmirakl &mdash; dual-GPU multi-agent simulation platform</footer>

</div>

<script>
var plotlyConfig = {responsive: true, displayModeBar: false};
{% if fan_chart_json %}
var fc = {{ fan_chart_json }};
Plotly.newPlot('fan-chart', fc.data, fc.layout, plotlyConfig);
{% endif %}
{% if spaghetti_json %}
var sp = {{ spaghetti_json }};
Plotly.newPlot('spaghetti-plot', sp.data, sp.layout, plotlyConfig);
{% endif %}
{% if tornado_json %}
var tn = {{ tornado_json }};
Plotly.newPlot('tornado-chart', tn.data, tn.layout, plotlyConfig);
{% endif %}
{% if variance_pie_json %}
var vp = {{ variance_pie_json }};
Plotly.newPlot('variance-pie', vp.data, vp.layout, plotlyConfig);
{% endif %}
{% if convergence_json %}
var cv = {{ convergence_json }};
Plotly.newPlot('convergence-plot', cv.data, cv.layout, plotlyConfig);
{% endif %}
</script>
</body>
</html>""")


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(
    ensemble_result: Optional[dict] = None,
    dynamics_analysis: Optional[dict] = None,
    scenario_tree: Optional[dict] = None,
    calibration_result: Optional[dict] = None,
    observed_data: Optional[dict] = None,
    title: str = "dualmirakl Simulation Report",
) -> str:
    """
    Generate a self-contained HTML report from simulation output.

    All arguments accept dicts (from .to_dict() or JSON). Charts use
    Plotly.js loaded from CDN — no local install needed.

    Args:
        ensemble_result: From EnsembleResult.to_dict() or NestedEnsembleResult.to_dict().
        dynamics_analysis: From analyze_simulation() output.
        scenario_tree: From tree_to_dict().
        calibration_result: From ABCResult.to_dict() or calibration_pipeline().
        observed_data: {metric: value} for scoring.
        title: Report title.

    Returns:
        Self-contained HTML string.
    """
    ctx: dict = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "experiment_id": "",
        "n_completed": 0,
        "n_failed": 0,
        "convergence_achieved": False,
        "final_cv": "N/A",
        "variance_decomposition": None,
        "pct_epistemic": "N/A",
        "pct_within": "N/A",
        "scoring_metrics": {},
        "fan_chart_json": None,
        "spaghetti_json": None,
        "tornado_json": None,
        "variance_pie_json": None,
        "convergence_json": None,
        "scenario_tree": None,
        "scenario_branches": [],
        "calibration": None,
        "calibration_params": [],
    }

    # ── Ensemble data ──────────────────────────────────────────────────
    if ensemble_result:
        ctx["experiment_id"] = ensemble_result.get("experiment_id", "")
        ctx["n_completed"] = ensemble_result.get("n_runs_completed", ensemble_result.get("n_completed", 0))
        ctx["n_failed"] = ensemble_result.get("n_runs_failed", ensemble_result.get("n_failed", 0))

        conv = ensemble_result.get("convergence", {})
        ctx["convergence_achieved"] = conv.get("achieved", False)
        ctx["final_cv"] = conv.get("final_cv", "N/A")

        summary = ensemble_result.get("ensemble_summary", {})
        bands = summary.get("percentile_bands", {})
        if bands:
            ctx["fan_chart_json"] = _fan_chart_json(bands)

        metric_vals = summary.get("metric_values", [])
        if metric_vals:
            threshold = conv.get("threshold", 0.05)
            ctx["convergence_json"] = _convergence_plot_json(metric_vals, threshold)

        # Spaghetti plot from all_score_logs
        logs = ensemble_result.get("all_score_logs", [])
        if logs:
            ctx["spaghetti_json"] = _spaghetti_plot_json(logs)

        # Variance decomposition (nested ensemble)
        vd = ensemble_result.get("variance_decomposition", {})
        if vd and vd.get("var_total", 0) > 0:
            ctx["variance_decomposition"] = vd
            ctx["pct_epistemic"] = round(vd.get("pct_epistemic", 0) * 100, 1)
            ctx["pct_within"] = round(vd.get("pct_within", 0) * 100, 1)
            ctx["variance_pie_json"] = _variance_pie_json(vd)

    # ── Dynamics / sensitivity ─────────────────────────────────────────
    if dynamics_analysis:
        sobol = dynamics_analysis.get("D_sobol_s2", {})
        if sobol:
            ctx["tornado_json"] = _tornado_chart_json(sobol)

    # ── Scoring ────────────────────────────────────────────────────────
    if observed_data and ensemble_result:
        try:
            from stats.scoring import score_ensemble
            summary = ensemble_result.get("ensemble_summary", {})
            metric_vals = summary.get("metric_values", [])
            if metric_vals:
                forecasts = {"mean_score": np.array(metric_vals)}
                scores = score_ensemble(forecasts, observed_data)
                for metric, score_dict in scores.items():
                    for score_name, val in score_dict.items():
                        ctx["scoring_metrics"][f"{metric}.{score_name}"] = val
        except Exception as e:
            logger.debug("Scoring failed: %s", e)

    # ── Scenario tree ──────────────────────────────────────────────────
    if scenario_tree:
        ctx["scenario_tree"] = scenario_tree
        branches = []

        def _flatten(node, depth=0):
            entry = {
                "id": ("  " * depth) + node.get("id", "?"),
                "probability": round(node.get("probability", 0) * 100, 1),
                "mean": node.get("metrics", {}).get("mean", "N/A"),
                "std": node.get("metrics", {}).get("std", "N/A"),
                "n_runs": node.get("n_supporting_runs", 0),
            }
            branches.append(entry)
            for child in node.get("children", []):
                _flatten(child, depth + 1)

        _flatten(scenario_tree)
        ctx["scenario_branches"] = branches

    # ── Calibration ────────────────────────────────────────────────────
    if calibration_result:
        abc = calibration_result.get("abc_result", calibration_result)
        ctx["calibration"] = abc
        names = abc.get("param_names", [])
        means = abc.get("posterior_mean", [])
        stds = abc.get("posterior_std", [])
        ctx["calibration_params"] = list(zip(names, means, stds))

    return REPORT_TEMPLATE.render(**ctx)

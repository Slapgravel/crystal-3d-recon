"""
dashboard.py — Live web dashboard for crystal growth monitoring.

Serves a local web page at http://localhost:5050 showing:
  - Real-time growth rate per facet (line chart)
  - Facet distance from centroid over time (line chart)
  - Experiment status (run count, elapsed time, next run countdown)
  - Representative image from the latest run
  - Downloadable CSV export of all measurements

The dashboard auto-refreshes every 30 seconds. It runs in a background
thread so it doesn't block the capture/reconstruct loop.

Usage:
    from crystal_recon.dashboard import Dashboard
    dash = Dashboard(db, experiment_name="crystal_KDP_001")
    dash.start()   # starts background thread
    # ... run experiment ...
    dash.stop()
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Dashboard HTML template — self-contained, no CDN dependencies for offline use
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<title>Crystal Growth Monitor — {experiment_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f1117; color: #e0e0e0; padding: 20px; }}
  h1 {{ font-size: 1.4rem; color: #7eb8f7; margin-bottom: 4px; }}
  .subtitle {{ font-size: 0.85rem; color: #888; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
           gap: 16px; margin-bottom: 20px; }}
  .card {{ background: #1a1d27; border-radius: 8px; padding: 16px;
           border: 1px solid #2a2d3a; }}
  .card h2 {{ font-size: 0.9rem; color: #aaa; margin-bottom: 12px;
              text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat {{ font-size: 2rem; font-weight: 600; color: #7eb8f7; }}
  .stat-label {{ font-size: 0.8rem; color: #666; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; padding: 6px 10px; color: #888;
        border-bottom: 1px solid #2a2d3a; font-weight: 500; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #1e2130; }}
  .positive {{ color: #4ade80; }}
  .negative {{ color: #f87171; }}
  .neutral {{ color: #888; }}
  canvas {{ width: 100% !important; }}
  .img-preview {{ max-width: 100%; border-radius: 4px; margin-top: 8px; }}
  .download-btn {{
    display: inline-block; margin-top: 12px; padding: 6px 14px;
    background: #2a2d3a; border-radius: 4px; color: #7eb8f7;
    text-decoration: none; font-size: 0.8rem;
  }}
  .download-btn:hover {{ background: #3a3d4a; }}
  .refresh-note {{ font-size: 0.75rem; color: #555; margin-top: 20px; }}
</style>
</head>
<body>
<h1>Crystal Growth Monitor</h1>
<div class="subtitle">{experiment_name} &nbsp;|&nbsp; Started: {start_time}
  &nbsp;|&nbsp; Elapsed: {elapsed} &nbsp;|&nbsp; Runs: {run_count}
  &nbsp;|&nbsp; Next run: ~{next_run_mins} min</div>

<div class="grid">
  <div class="card">
    <h2>Max Growth Rate</h2>
    <div class="stat">{max_rate}</div>
    <div class="stat-label">mm / hr (latest)</div>
  </div>
  <div class="card">
    <h2>Active Facets</h2>
    <div class="stat">{facet_count}</div>
    <div class="stat-label">tracked facets</div>
  </div>
  <div class="card">
    <h2>Total Runs</h2>
    <div class="stat">{run_count}</div>
    <div class="stat-label">completed reconstructions</div>
  </div>
</div>

<div class="grid">
  <div class="card" style="grid-column: span 2;">
    <h2>Facet Growth Rates</h2>
    <table>
      <thead>
        <tr><th>Facet</th><th>Latest Rate (mm/hr)</th>
            <th>Mean Rate (mm/hr)</th><th>Total Growth (mm)</th><th>Measurements</th></tr>
      </thead>
      <tbody>
        {facet_rows}
      </tbody>
    </table>
    <a href="/export/csv" class="download-btn">Download CSV</a>
  </div>

  {image_card}
</div>

<div class="card">
  <h2>Growth Rate History (all facets)</h2>
  <canvas id="rateChart" height="120"></canvas>
</div>

<div class="card" style="margin-top: 16px;">
  <h2>Facet Distance from Centroid</h2>
  <canvas id="distChart" height="120"></canvas>
</div>

<p class="refresh-note">Page auto-refreshes every 30 seconds.</p>

<script src="/static/chart.min.js"></script>
<script>
const chartData = {chart_data_json};

function makeChart(id, datasets, yLabel) {{
  const ctx = document.getElementById(id);
  if (!ctx || !datasets.length) return;
  const colors = ['#7eb8f7','#4ade80','#f87171','#fbbf24','#a78bfa','#34d399','#fb923c','#e879f9'];
  new Chart(ctx, {{
    type: 'line',
    data: {{
      datasets: datasets.map((d, i) => ({{
        label: d.label,
        data: d.points,
        borderColor: colors[i % colors.length],
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 2,
        tension: 0.3,
      }}))
    }},
    options: {{
      responsive: true,
      parsing: {{ xAxisKey: 'x', yAxisKey: 'y' }},
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'Elapsed (hr)', color: '#888' }},
              ticks: {{ color: '#888' }}, grid: {{ color: '#2a2d3a' }} }},
        y: {{ title: {{ display: true, text: yLabel, color: '#888' }},
              ticks: {{ color: '#888' }}, grid: {{ color: '#2a2d3a' }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }} }}
    }}
  }});
}}

const t0 = chartData.t0;
makeChart('rateChart',
  chartData.growth_rates.map(d => ({{
    label: d.facet_id,
    points: d.data.map(p => ({{ x: (p.ts - t0) / 3600, y: p.rate }}))
  }})),
  'Growth Rate (mm/hr)'
);
makeChart('distChart',
  chartData.distances.map(d => ({{
    label: d.facet_id,
    points: d.data.map(p => ({{ x: (p.ts - t0) / 3600, y: p.dist }}))
  }})),
  'Distance from Centroid (mm)'
);
</script>
</body>
</html>
"""


class Dashboard:
    """
    Live web dashboard for crystal growth monitoring.

    Runs a lightweight Flask server in a background thread.
    """

    def __init__(self, db, experiment_name: str, port: int = 5050):
        """
        Args:
            db:               GrowthDatabase instance.
            experiment_name:  Displayed in the dashboard title.
            port:             Local port to serve on (default 5050).
        """
        self.db = db
        self.experiment_name = experiment_name
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._next_run_mins: int = 0
        self._app = None

    def set_next_run_mins(self, mins: int):
        """Update the countdown to the next run (called by the scheduler)."""
        self._next_run_mins = mins

    def start(self, start_time: Optional[float] = None):
        """Start the dashboard server in a background thread."""
        import time as _time
        self._start_time = start_time or _time.time()
        self._build_app()
        self._thread = threading.Thread(
            target=self._run_server, daemon=True, name="dashboard"
        )
        self._thread.start()
        logger.info(f"Dashboard running at http://localhost:{self.port}")

    def stop(self):
        """Stop the dashboard server (daemon thread will exit with main process)."""
        logger.info("Dashboard stopped.")

    def _build_app(self):
        """Build the Flask application."""
        try:
            from flask import Flask, Response, send_file
        except ImportError:
            logger.warning("Flask not installed — dashboard disabled. "
                           "Install with: pip install flask")
            return

        app = Flask(__name__, static_folder=None)
        # Suppress Flask startup banner (set log level, not WERKZEUG_RUN_MAIN)
        import os
        import logging as _flask_log
        _flask_log.getLogger("werkzeug").setLevel(_flask_log.ERROR)

        # Serve Chart.js from a local copy if available, else CDN fallback
        chart_js_path = Path(__file__).parent.parent / "static" / "chart.min.js"

        @app.route("/static/chart.min.js")
        def serve_chartjs():
            if chart_js_path.exists():
                return send_file(str(chart_js_path), mimetype="application/javascript")
            # Fallback: redirect to CDN
            from flask import redirect
            return redirect("https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js")

        @app.route("/")
        def index():
            return Response(self._render_html(), mimetype="text/html")

        @app.route("/data")
        def data():
            return Response(
                json.dumps(self._build_chart_data()),
                mimetype="application/json"
            )

        @app.route("/export/csv")
        def export_csv():
            csv = self._build_csv()
            return Response(
                csv,
                mimetype="text/csv",
                headers={"Content-Disposition":
                         f"attachment; filename={self.experiment_name}_growth.csv"}
            )

        @app.route("/image/latest")
        def image_latest():
            """Serve the representative image from the most recent run."""
            runs = self.db.get_runs()
            if not runs:
                from flask import abort
                abort(404)
            latest_img = runs[-1].get("representative_image")
            if not latest_img or not Path(latest_img).exists():
                from flask import abort
                abort(404)
            return send_file(str(Path(latest_img).resolve()),
                             mimetype="image/jpeg")

        self._app = app

    def _run_server(self):
        """Run the Flask development server (blocking)."""
        if self._app is None:
            return
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
        self._app.run(host="0.0.0.0", port=self.port, debug=False, use_reloader=False)

    def _render_html(self) -> str:
        """Render the dashboard HTML with current data."""
        import time as _time
        from datetime import datetime

        data = self.db.get_dashboard_data()
        runs = data["runs"]
        facet_ids = data["facet_ids"]
        growth_rates = data["growth_rates"]
        distances = data["distances"]

        run_count = len(runs)
        start_ts = runs[0]["timestamp"] if runs else _time.time()
        elapsed_hrs = (_time.time() - start_ts) / 3600

        # Elapsed string
        h = int(elapsed_hrs)
        m = int((elapsed_hrs - h) * 60)
        elapsed_str = f"{h}h {m}m"

        start_str = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M")

        # Max growth rate
        all_rates = [
            r["rate"] for fid in facet_ids for r in growth_rates.get(fid, [])
        ]
        max_rate = f"{max(all_rates):.4f}" if all_rates else "—"

        # Facet table rows
        rows = []
        for fid in facet_ids:
            gr = growth_rates.get(fid, [])
            dist = distances.get(fid, [])
            latest_rate = gr[-1]["rate"] if gr else None
            mean_rate = sum(r["rate"] for r in gr) / len(gr) if gr else None
            total_growth = (dist[-1]["dist"] - dist[0]["dist"]) if len(dist) >= 2 else None
            n = len(dist)

            def fmt_rate(v):
                if v is None:
                    return '<td class="neutral">—</td>'
                cls = "positive" if v >= 0 else "negative"
                return f'<td class="{cls}">{v:+.4f}</td>'

            rows.append(
                f"<tr><td>{fid}</td>"
                f"{fmt_rate(latest_rate)}"
                f"{fmt_rate(mean_rate)}"
                f"<td>{'—' if total_growth is None else f'{total_growth:.3f}'}</td>"
                f"<td>{n}</td></tr>"
            )

        # Representative image card
        latest_img = None
        if runs:
            latest_img = runs[-1].get("representative_image")
        if latest_img and Path(latest_img).exists():
            image_card = (
                f'<div class="card"><h2>Latest Image ({runs[-1]["datetime_str"]})</h2>'
                f'<img src="/image/latest" class="img-preview" alt="Latest crystal image"></div>'
            )
        else:
            image_card = ""

        # Chart data
        chart_data = json.dumps(self._build_chart_data())

        return _HTML_TEMPLATE.format(
            experiment_name=self.experiment_name,
            start_time=start_str,
            elapsed=elapsed_str,
            run_count=run_count,
            next_run_mins=self._next_run_mins,
            max_rate=max_rate,
            facet_count=len(facet_ids),
            facet_rows="\n".join(rows) if rows else "<tr><td colspan='5'>No data yet</td></tr>",
            image_card=image_card,
            chart_data_json=chart_data,
        )

    def _build_chart_data(self) -> dict:
        """Build the JSON data structure for Chart.js."""
        data = self.db.get_dashboard_data()
        runs = data["runs"]
        t0 = runs[0]["timestamp"] if runs else 0

        return {
            "t0": t0,
            "growth_rates": [
                {"facet_id": fid,
                 "data": data["growth_rates"].get(fid, [])}
                for fid in data["facet_ids"]
            ],
            "distances": [
                {"facet_id": fid,
                 "data": data["distances"].get(fid, [])}
                for fid in data["facet_ids"]
            ],
        }

    def _build_csv(self) -> str:
        """Build a CSV export of all growth rate measurements."""
        data = self.db.get_dashboard_data()
        lines = ["facet_id,timestamp,datetime,distance_mm,growth_rate_mm_hr"]
        for fid in data["facet_ids"]:
            dists = {d["ts"]: d["dist"] for d in data["distances"].get(fid, [])}
            rates = {r["ts"]: r["rate"] for r in data["growth_rates"].get(fid, [])}
            for ts, dist in sorted(dists.items()):
                dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                rate = rates.get(ts, "")
                lines.append(f"{fid},{ts:.1f},{dt_str},{dist:.4f},{rate}")
        return "\n".join(lines) + "\n"

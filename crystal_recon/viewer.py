"""
viewer.py — Standalone desktop GUI for crystal growth experiment data.

Reads directly from the SQLite database (.db) file produced by monitor.py.
No network socket is required: point it at a .db file on a local drive,
a network share, or a USB stick and it works identically.

Usage
-----
From the command line:
    python -m crystal_recon.viewer                        # opens file picker
    python -m crystal_recon.viewer output/my_exp.db      # opens directly

As a module:
    from crystal_recon.viewer import launch
    launch("output/my_exp.db")

Dependencies
------------
    PyQt5       (pip install PyQt5)
    matplotlib  (pip install matplotlib)
    numpy       (already required by the project)

The viewer auto-refreshes every 60 seconds when the database file is on a
network drive, so you can watch a live experiment from any machine that can
read the share.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy Qt / matplotlib imports (so the module can be imported without a
# display server present, e.g. during unit tests)
# ---------------------------------------------------------------------------

def _require_qt():
    try:
        from PyQt5 import QtWidgets, QtCore, QtGui
        return QtWidgets, QtCore, QtGui
    except ImportError:
        print("PyQt5 is required for the viewer.\n"
              "Install it with:  pip install PyQt5", file=sys.stderr)
        sys.exit(1)


def _require_mpl():
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    return plt, FigureCanvas, Figure


# ---------------------------------------------------------------------------
# Thin read-only database wrapper (no dependency on crystal_recon.config)
# ---------------------------------------------------------------------------

class _DB:
    """Read-only view of a GrowthDatabase SQLite file."""

    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def reload(self):
        """Re-open the connection so we see any new rows written since last read."""
        self._conn.close()
        self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True,
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def runs(self) -> list[dict]:
        return [dict(r) for r in
                self._conn.execute("SELECT * FROM runs ORDER BY timestamp").fetchall()]

    def facet_ids(self) -> list[str]:
        return [r[0] for r in
                self._conn.execute(
                    "SELECT DISTINCT facet_id FROM facets ORDER BY facet_id"
                ).fetchall()]

    def distances(self, facet_id: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT f.distance_mm, r.timestamp
               FROM facets f JOIN runs r ON f.run_id = r.run_id
               WHERE f.facet_id = ?
               ORDER BY r.timestamp""",
            (facet_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def growth_rates(self, facet_id: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT gr.rate_mm_hr, gr.dt_hours, gr.dd_mm,
                      r.timestamp as ts_to
               FROM growth_rates gr
               JOIN runs r ON gr.run_id_to = r.run_id
               WHERE gr.facet_id = ?
               ORDER BY r.timestamp""",
            (facet_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        runs = self.runs()
        fids = self.facet_ids()
        all_rates = [
            r["rate_mm_hr"]
            for fid in fids
            for r in self.growth_rates(fid)
        ]
        return {
            "run_count": len(runs),
            "facet_count": len(fids),
            "start_ts": runs[0]["timestamp"] if runs else None,
            "latest_ts": runs[-1]["timestamp"] if runs else None,
            "latest_img": runs[-1].get("representative_image") if runs else None,
            "max_rate": max(all_rates) if all_rates else None,
            "mean_rate": float(np.mean(all_rates)) if all_rates else None,
        }

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Colour palette (matches the dark web dashboard)
# ---------------------------------------------------------------------------

_PALETTE = [
    "#7eb8f7", "#4ade80", "#f87171", "#fbbf24",
    "#a78bfa", "#34d399", "#fb923c", "#e879f9",
    "#38bdf8", "#86efac", "#fca5a5", "#fde68a",
]

_BG   = "#0f1117"
_CARD = "#1a1d27"
_GRID = "#2a2d3a"
_TEXT = "#e0e0e0"
_SUB  = "#888888"


# ---------------------------------------------------------------------------
# Matplotlib chart widget
# ---------------------------------------------------------------------------

class _ChartWidget:
    """A matplotlib Figure embedded in a Qt widget."""

    def __init__(self, title: str, ylabel: str, parent=None):
        _, FigureCanvas, Figure = _require_mpl()
        self.fig = Figure(figsize=(6, 3), facecolor=_BG, tight_layout=True)
        self.ax  = self.fig.add_subplot(111, facecolor=_CARD)
        self.ax.set_title(title, color=_TEXT, fontsize=9, pad=6)
        self.ax.set_xlabel("Elapsed (hr)", color=_SUB, fontsize=8)
        self.ax.set_ylabel(ylabel, color=_SUB, fontsize=8)
        self.ax.tick_params(colors=_SUB, labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(_GRID)
        self.ax.grid(color=_GRID, linewidth=0.5)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(220)

    def plot(self, series: dict[str, list[tuple[float, float]]]):
        """
        series: {facet_id: [(elapsed_hr, value), ...]}
        """
        self.ax.cla()
        self.ax.set_facecolor(_CARD)
        self.ax.tick_params(colors=_SUB, labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(_GRID)
        self.ax.grid(color=_GRID, linewidth=0.5)

        for i, (fid, pts) in enumerate(series.items()):
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            colour = _PALETTE[i % len(_PALETTE)]
            self.ax.plot(xs, ys, "o-", color=colour, linewidth=1.5,
                         markersize=3, label=fid)

        if series:
            leg = self.ax.legend(fontsize=7, facecolor=_CARD,
                                 edgecolor=_GRID, labelcolor=_TEXT,
                                 loc="best")
            leg.get_frame().set_alpha(0.8)

        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ViewerWindow:
    """
    The main application window.

    Layout
    ------
    ┌─────────────────────────────────────────────────────┐
    │  [Open DB]  path/to/experiment.db  [Refresh]  [●live]│  toolbar
    ├──────────────────────┬──────────────────────────────┤
    │  Experiment summary  │  Latest crystal image        │  top row
    ├──────────────────────┴──────────────────────────────┤
    │  Growth rate (mm/hr) over time  [line chart]        │
    ├─────────────────────────────────────────────────────┤
    │  Facet distance from centroid   [line chart]        │
    ├─────────────────────────────────────────────────────┤
    │  Facet table  (id | latest rate | mean | Δdist | n) │
    └─────────────────────────────────────────────────────┘
    """

    REFRESH_MS = 60_000   # auto-refresh interval when live mode is on

    def __init__(self, db_path: Optional[str] = None):
        QtWidgets, QtCore, QtGui = _require_qt()
        _require_mpl()

        self._db: Optional[_DB] = None
        self._timer = None

        # --- Main window ---
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Crystal Growth Viewer")
        self.win.resize(1100, 820)
        self._apply_dark_style(QtWidgets)

        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Toolbar ---
        toolbar = QtWidgets.QHBoxLayout()
        self._btn_open = QtWidgets.QPushButton("📂  Open database…")
        self._btn_open.setFixedHeight(30)
        self._btn_open.clicked.connect(self._on_open)

        self._lbl_path = QtWidgets.QLabel("No database loaded")
        self._lbl_path.setStyleSheet(f"color: {_SUB}; font-size: 11px;")

        self._btn_refresh = QtWidgets.QPushButton("↻  Refresh")
        self._btn_refresh.setFixedHeight(30)
        self._btn_refresh.setEnabled(False)
        self._btn_refresh.clicked.connect(self._refresh)

        self._chk_live = QtWidgets.QCheckBox("Auto-refresh (60 s)")
        self._chk_live.setStyleSheet(f"color: {_TEXT};")
        self._chk_live.stateChanged.connect(self._on_live_toggle)

        toolbar.addWidget(self._btn_open)
        toolbar.addWidget(self._lbl_path, stretch=1)
        toolbar.addWidget(self._btn_refresh)
        toolbar.addWidget(self._chk_live)
        root.addLayout(toolbar)

        # --- Top row: summary + image ---
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(6)

        # Summary card
        summary_frame = self._card_frame(QtWidgets)
        summary_layout = QtWidgets.QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(10, 8, 10, 8)
        self._summary_labels: dict[str, QtWidgets.QLabel] = {}
        for key in ("Experiment", "Runs completed", "Facets tracked",
                    "Started", "Latest run", "Elapsed",
                    "Max growth rate", "Mean growth rate"):
            row = QtWidgets.QHBoxLayout()
            lbl_key = QtWidgets.QLabel(key + ":")
            lbl_key.setStyleSheet(f"color: {_SUB}; font-size: 11px; min-width: 130px;")
            lbl_val = QtWidgets.QLabel("—")
            lbl_val.setStyleSheet(f"color: {_TEXT}; font-size: 11px; font-weight: bold;")
            row.addWidget(lbl_key)
            row.addWidget(lbl_val, stretch=1)
            summary_layout.addLayout(row)
            self._summary_labels[key] = lbl_val
        summary_layout.addStretch()
        top_row.addWidget(summary_frame, stretch=1)

        # Image card
        img_frame = self._card_frame(QtWidgets)
        img_layout = QtWidgets.QVBoxLayout(img_frame)
        img_layout.setContentsMargins(8, 8, 8, 8)
        img_lbl_title = QtWidgets.QLabel("Latest representative image")
        img_lbl_title.setStyleSheet(
            f"color: {_SUB}; font-size: 9px; text-transform: uppercase; "
            f"letter-spacing: 1px;"
        )
        self._img_label = QtWidgets.QLabel()
        self._img_label.setAlignment(QtCore.Qt.AlignCenter)
        self._img_label.setMinimumSize(320, 180)
        self._img_label.setStyleSheet(f"color: {_SUB}; font-size: 11px;")
        self._img_label.setText("No image available")
        img_layout.addWidget(img_lbl_title)
        img_layout.addWidget(self._img_label, stretch=1)
        top_row.addWidget(img_frame, stretch=1)

        root.addLayout(top_row)

        # --- Charts ---
        self._rate_chart = _ChartWidget("Growth Rate per Facet", "mm / hr")
        self._dist_chart = _ChartWidget("Facet Distance from Centroid", "mm")

        charts_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        charts_splitter.setStyleSheet("QSplitter::handle { background: #2a2d3a; }")
        charts_splitter.addWidget(self._rate_chart.canvas)
        charts_splitter.addWidget(self._dist_chart.canvas)
        charts_splitter.setSizes([240, 240])
        root.addWidget(charts_splitter, stretch=2)

        # --- Facet table ---
        table_frame = self._card_frame(QtWidgets)
        table_layout = QtWidgets.QVBoxLayout(table_frame)
        table_layout.setContentsMargins(8, 6, 8, 6)
        tbl_title = QtWidgets.QLabel("Facet summary")
        tbl_title.setStyleSheet(
            f"color: {_SUB}; font-size: 9px; text-transform: uppercase; "
            f"letter-spacing: 1px; margin-bottom: 4px;"
        )
        table_layout.addWidget(tbl_title)

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Facet", "Latest rate (mm/hr)", "Mean rate (mm/hr)",
             "Δ distance (mm)", "Measurements", "Last seen"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            f"QTableWidget {{ background: {_CARD}; color: {_TEXT}; "
            f"  gridline-color: {_GRID}; font-size: 11px; border: none; }}"
            f"QTableWidget::item:alternate {{ background: #12151f; }}"
            f"QHeaderView::section {{ background: {_BG}; color: {_SUB}; "
            f"  font-size: 10px; border: 1px solid {_GRID}; padding: 3px; }}"
            f"QTableWidget::item:selected {{ background: #2a3a5a; }}"
        )
        self._table.setMaximumHeight(180)
        table_layout.addWidget(self._table)
        root.addWidget(table_frame)

        # --- Status bar ---
        self._status = self.win.statusBar()
        self._status.setStyleSheet(f"color: {_SUB}; font-size: 10px;")
        self._status.showMessage("Ready — open a .db file to begin.")

        # --- Auto-refresh timer ---
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._refresh)

        # Load immediately if path given
        if db_path:
            self._load(db_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _card_frame(QtWidgets):
        f = QtWidgets.QFrame()
        f.setStyleSheet(
            f"QFrame {{ background: {_CARD}; border-radius: 6px; "
            f"border: 1px solid {_GRID}; }}"
        )
        return f

    @staticmethod
    def _apply_dark_style(QtWidgets):
        app = QtWidgets.QApplication.instance()
        app.setStyle("Fusion")
        from PyQt5.QtGui import QPalette, QColor
        pal = QPalette()
        pal.setColor(QPalette.Window,          QColor(_BG))
        pal.setColor(QPalette.WindowText,      QColor(_TEXT))
        pal.setColor(QPalette.Base,            QColor(_CARD))
        pal.setColor(QPalette.AlternateBase,   QColor("#12151f"))
        pal.setColor(QPalette.Text,            QColor(_TEXT))
        pal.setColor(QPalette.Button,          QColor("#252836"))
        pal.setColor(QPalette.ButtonText,      QColor(_TEXT))
        pal.setColor(QPalette.Highlight,       QColor("#2a3a5a"))
        pal.setColor(QPalette.HighlightedText, QColor(_TEXT))
        pal.setColor(QPalette.ToolTipBase,     QColor(_CARD))
        pal.setColor(QPalette.ToolTipText,     QColor(_TEXT))
        app.setPalette(pal)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_open(self):
        QtWidgets, QtCore, QtGui = _require_qt()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.win,
            "Open experiment database",
            str(Path.home()),
            "SQLite databases (*.db);;All files (*)"
        )
        if path:
            self._load(path)

    def _on_live_toggle(self, state):
        if state and self._db:
            self._timer.start(self.REFRESH_MS)
            self._status.showMessage(
                f"Live mode — refreshing every {self.REFRESH_MS // 1000} s"
            )
        else:
            self._timer.stop()
            if self._db:
                self._status.showMessage("Live mode off.")

    def _load(self, path: str):
        if self._db:
            self._db.close()
        try:
            self._db = _DB(path)
        except Exception as e:
            QtWidgets, _, _ = _require_qt()
            QtWidgets.QMessageBox.critical(
                self.win, "Cannot open database",
                f"Failed to open:\n{path}\n\n{e}"
            )
            return
        self._lbl_path.setText(str(path))
        self._btn_refresh.setEnabled(True)
        self._refresh()
        if self._chk_live.isChecked():
            self._timer.start(self.REFRESH_MS)

    def _refresh(self):
        if self._db is None:
            return
        try:
            self._db.reload()
            self._update_ui()
            self._status.showMessage(
                f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}"
            )
        except Exception as e:
            self._status.showMessage(f"Refresh error: {e}")

    # ------------------------------------------------------------------
    # UI update
    # ------------------------------------------------------------------

    def _update_ui(self):
        db = self._db
        runs   = db.runs()
        fids   = db.facet_ids()
        summ   = db.summary()

        # Experiment name from the db file stem
        exp_name = Path(db.path).stem

        # --- Summary labels ---
        def _ts(ts):
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"

        def _rate(v):
            return f"{v:+.4f} mm/hr" if v is not None else "—"

        elapsed_str = "—"
        if summ["start_ts"] and summ["latest_ts"]:
            delta = summ["latest_ts"] - summ["start_ts"]
            h = int(delta // 3600)
            m = int((delta % 3600) // 60)
            elapsed_str = f"{h}h {m}m"

        self._summary_labels["Experiment"].setText(exp_name)
        self._summary_labels["Runs completed"].setText(str(summ["run_count"]))
        self._summary_labels["Facets tracked"].setText(str(summ["facet_count"]))
        self._summary_labels["Started"].setText(_ts(summ["start_ts"]))
        self._summary_labels["Latest run"].setText(_ts(summ["latest_ts"]))
        self._summary_labels["Elapsed"].setText(elapsed_str)
        self._summary_labels["Max growth rate"].setText(_rate(summ["max_rate"]))
        self._summary_labels["Mean growth rate"].setText(_rate(summ["mean_rate"]))

        # --- Representative image ---
        self._update_image(summ.get("latest_img"))

        # --- Charts ---
        t0 = summ["start_ts"] or 0.0

        rate_series: dict[str, list] = {}
        dist_series: dict[str, list] = {}

        for fid in fids:
            gr = db.growth_rates(fid)
            rate_series[fid] = [
                ((r["ts_to"] - t0) / 3600, r["rate_mm_hr"]) for r in gr
            ]
            dists = db.distances(fid)
            dist_series[fid] = [
                ((d["timestamp"] - t0) / 3600, d["distance_mm"]) for d in dists
            ]

        self._rate_chart.plot(rate_series)
        self._dist_chart.plot(dist_series)

        # --- Facet table ---
        QtWidgets, _, _ = _require_qt()
        self._table.setRowCount(len(fids))

        for row, fid in enumerate(fids):
            gr    = db.growth_rates(fid)
            dists = db.distances(fid)

            latest_rate  = gr[-1]["rate_mm_hr"] if gr else None
            mean_rate    = float(np.mean([r["rate_mm_hr"] for r in gr])) if gr else None
            delta_dist   = (dists[-1]["distance_mm"] - dists[0]["distance_mm"]
                            ) if len(dists) >= 2 else None
            n            = len(dists)
            last_seen    = _ts(dists[-1]["timestamp"]) if dists else "—"

            def _cell(text, align=None, colour=None):
                item = QtWidgets.QTableWidgetItem(text)
                if align:
                    item.setTextAlignment(align)
                if colour:
                    from PyQt5.QtGui import QColor
                    item.setForeground(QColor(colour))
                return item

            def _fmt_rate(v):
                if v is None:
                    return "—", None
                col = "#4ade80" if v >= 0 else "#f87171"
                return f"{v:+.4f}", col

            lr_txt, lr_col = _fmt_rate(latest_rate)
            mr_txt, mr_col = _fmt_rate(mean_rate)
            dd_txt = f"{delta_dist:+.3f}" if delta_dist is not None else "—"

            Qt = __import__("PyQt5.QtCore", fromlist=["Qt"]).Qt
            self._table.setItem(row, 0, _cell(fid))
            self._table.setItem(row, 1, _cell(lr_txt,
                Qt.AlignRight | Qt.AlignVCenter, lr_col))
            self._table.setItem(row, 2, _cell(mr_txt,
                Qt.AlignRight | Qt.AlignVCenter, mr_col))
            self._table.setItem(row, 3, _cell(dd_txt,
                Qt.AlignRight | Qt.AlignVCenter))
            self._table.setItem(row, 4, _cell(str(n),
                Qt.AlignRight | Qt.AlignVCenter))
            self._table.setItem(row, 5, _cell(last_seen))

        self._table.resizeColumnsToContents()

    def _update_image(self, img_path: Optional[str]):
        QtWidgets, QtCore, QtGui = _require_qt()
        if not img_path or not Path(img_path).exists():
            self._img_label.setText("No image available")
            self._img_label.setPixmap(QtGui.QPixmap())
            return
        pixmap = QtGui.QPixmap(str(img_path))
        if pixmap.isNull():
            self._img_label.setText(f"Cannot load image:\n{img_path}")
            return
        scaled = pixmap.scaled(
            self._img_label.width() - 8,
            self._img_label.height() - 8,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self._img_label.setPixmap(scaled)
        self._img_label.setText("")

    def show(self):
        self.win.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch(db_path: Optional[str] = None):
    """Create the Qt application and open the viewer window."""
    QtWidgets, _, _ = _require_qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = ViewerWindow(db_path)
    window.show()
    sys.exit(app.exec_())


def main():
    parser = argparse.ArgumentParser(
        description="Crystal Growth Viewer — reads a monitor.py .db file directly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m crystal_recon.viewer
  python -m crystal_recon.viewer output/crystal_KDP_001.db
  python -m crystal_recon.viewer Z:\\lab\\experiments\\crystal_KDP_001.db
        """,
    )
    parser.add_argument(
        "db",
        nargs="?",
        default=None,
        help="Path to the experiment .db file (opens a file picker if omitted).",
    )
    args = parser.parse_args()
    launch(args.db)


if __name__ == "__main__":
    main()

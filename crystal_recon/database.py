"""
database.py — SQLite data store for crystal growth measurements.

Stores all facet measurements, growth rates, and run metadata in a single
SQLite database file. This allows the experiment to be paused and resumed,
and provides a complete record of all measurements for later analysis.

Schema:
  runs        — one row per capture+reconstruct cycle
  facets      — one row per detected facet per run
  growth_rates — computed growth rates between consecutive runs

The database file is stored at: <OUTPUT_DIR>/<experiment_name>.db
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

from crystal_recon import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

class GrowthDatabase:
    """
    SQLite database for storing crystal growth measurements.

    Usage:
        db = GrowthDatabase("crystal_KDP_001")
        db.add_run("crystal_KDP_001_20260320_143000", timestamp, point_count)
        db.add_facet(run_id, facet_id, normal, distance, area_pts)
        db.add_growth_rate(facet_id, run_id_a, run_id_b, rate_mm_hr)
        rates = db.get_growth_rates("F0")
    """

    def __init__(self, experiment_name: str):
        """
        Open (or create) the database for the given experiment.

        Args:
            experiment_name: Used as the database filename stem.
        """
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = output_dir / f"{experiment_name}.db"
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Database opened: {self.db_path}")

    def _create_tables(self):
        """Create tables if they don't already exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id          TEXT PRIMARY KEY,
                timestamp       REAL NOT NULL,
                datetime_str    TEXT NOT NULL,
                point_count     INTEGER,
                representative_image TEXT,
                notes           TEXT
            );

            CREATE TABLE IF NOT EXISTS facets (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL REFERENCES runs(run_id),
                facet_id        TEXT NOT NULL,
                normal_x        REAL NOT NULL,
                normal_y        REAL NOT NULL,
                normal_z        REAL NOT NULL,
                distance_mm     REAL NOT NULL,
                area_pts        INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS growth_rates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                facet_id        TEXT NOT NULL,
                run_id_from     TEXT NOT NULL REFERENCES runs(run_id),
                run_id_to       TEXT NOT NULL REFERENCES runs(run_id),
                dt_hours        REAL NOT NULL,
                dd_mm           REAL NOT NULL,
                rate_mm_hr      REAL NOT NULL,
                computed_at     REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_facets_run ON facets(run_id);
            CREATE INDEX IF NOT EXISTS idx_facets_id ON facets(facet_id);
            CREATE INDEX IF NOT EXISTS idx_growth_facet ON growth_rates(facet_id);
        """)
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Write methods
    # -----------------------------------------------------------------------

    def add_run(self,
                run_id: str,
                timestamp: float,
                point_count: int = 0,
                representative_image: Optional[str] = None,
                notes: Optional[str] = None):
        """Record a completed reconstruction run."""
        from datetime import datetime
        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        self._conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, timestamp, datetime_str, point_count, representative_image, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, timestamp, dt_str, point_count, representative_image, notes)
        )
        self._conn.commit()
        logger.debug(f"Run recorded: {run_id}")

    def add_facet(self,
                  run_id: str,
                  facet_id: str,
                  normal: np.ndarray,
                  distance_mm: float,
                  area_pts: int):
        """Record a detected facet for a given run."""
        self._conn.execute(
            """INSERT INTO facets
               (run_id, facet_id, normal_x, normal_y, normal_z, distance_mm, area_pts)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, facet_id,
             float(normal[0]), float(normal[1]), float(normal[2]),
             float(distance_mm), int(area_pts))
        )
        self._conn.commit()

    def add_growth_rate(self,
                        facet_id: str,
                        run_id_from: str,
                        run_id_to: str,
                        dt_hours: float,
                        dd_mm: float,
                        rate_mm_hr: float):
        """Record a computed growth rate between two consecutive runs."""
        self._conn.execute(
            """INSERT INTO growth_rates
               (facet_id, run_id_from, run_id_to, dt_hours, dd_mm, rate_mm_hr, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (facet_id, run_id_from, run_id_to,
             float(dt_hours), float(dd_mm), float(rate_mm_hr), time.time())
        )
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Read methods
    # -----------------------------------------------------------------------

    def get_runs(self) -> list:
        """Return all runs ordered by timestamp."""
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY timestamp"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_facets_for_run(self, run_id: str) -> list:
        """Return all facets detected in a given run."""
        rows = self._conn.execute(
            "SELECT * FROM facets WHERE run_id = ? ORDER BY area_pts DESC",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_growth_rates(self, facet_id: str) -> list:
        """
        Return all growth rate measurements for a given facet,
        ordered by time.
        """
        rows = self._conn.execute(
            """SELECT gr.*, r.timestamp as ts_to
               FROM growth_rates gr
               JOIN runs r ON gr.run_id_to = r.run_id
               WHERE gr.facet_id = ?
               ORDER BY r.timestamp""",
            (facet_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_facet_ids(self) -> list:
        """Return the list of all unique facet IDs seen across all runs."""
        rows = self._conn.execute(
            "SELECT DISTINCT facet_id FROM facets ORDER BY facet_id"
        ).fetchall()
        return [r["facet_id"] for r in rows]

    def get_latest_distances(self) -> dict:
        """
        Return the most recent distance measurement for each facet.

        Returns:
            Dict mapping facet_id -> (distance_mm, timestamp, run_id)
        """
        rows = self._conn.execute(
            """SELECT f.facet_id, f.distance_mm, r.timestamp, r.run_id
               FROM facets f
               JOIN runs r ON f.run_id = r.run_id
               WHERE r.timestamp = (
                   SELECT MAX(r2.timestamp)
                   FROM facets f2
                   JOIN runs r2 ON f2.run_id = r2.run_id
                   WHERE f2.facet_id = f.facet_id
               )
               ORDER BY f.facet_id"""
        ).fetchall()
        return {r["facet_id"]: (r["distance_mm"], r["timestamp"], r["run_id"])
                for r in rows}

    def get_dashboard_data(self) -> dict:
        """
        Return all data needed to render the dashboard.

        Returns a dict with:
          - 'facet_ids': list of facet IDs
          - 'runs': list of run dicts
          - 'growth_rates': dict facet_id -> list of {ts, rate} dicts
          - 'distances': dict facet_id -> list of {ts, dist} dicts
        """
        facet_ids = self.get_all_facet_ids()
        runs = self.get_runs()

        growth_rates = {}
        distances = {}

        for fid in facet_ids:
            gr_rows = self.get_growth_rates(fid)
            growth_rates[fid] = [
                {"ts": r["ts_to"], "rate": r["rate_mm_hr"]} for r in gr_rows
            ]

            dist_rows = self._conn.execute(
                """SELECT f.distance_mm, r.timestamp
                   FROM facets f JOIN runs r ON f.run_id = r.run_id
                   WHERE f.facet_id = ?
                   ORDER BY r.timestamp""",
                (fid,)
            ).fetchall()
            distances[fid] = [
                {"ts": r["timestamp"], "dist": r["distance_mm"]} for r in dist_rows
            ]

        return {
            "facet_ids": facet_ids,
            "runs": runs,
            "growth_rates": growth_rates,
            "distances": distances,
        }

    def run_count(self) -> int:
        """Return the total number of completed runs."""
        return self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    def close(self):
        """Close the database connection."""
        self._conn.close()
        logger.info("Database closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

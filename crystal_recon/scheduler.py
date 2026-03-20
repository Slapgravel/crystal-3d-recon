"""
scheduler.py — Adaptive capture scheduling for crystal growth monitoring.

Manages the timing of repeated capture+reconstruct+analyse cycles, adapting
the interval based on measured growth rate. Also handles image cleanup after
each run, keeping only a single representative image per dataset.

Interval schedule (configurable in config.py):
  - Phase 1: every INTERVAL_PHASE1_MINS for the first PHASE1_HOURS hours
  - Phase 2: every INTERVAL_PHASE2_MINS for the next PHASE2_HOURS hours
  - Phase 3: every INTERVAL_PHASE3_MINS thereafter
  - Adaptive: if growth rate exceeds ADAPTIVE_FAST_THRESHOLD mm/hr,
    the interval is halved (down to INTERVAL_MIN_MINS minimum)
"""

import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from crystal_recon import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interval schedule
# ---------------------------------------------------------------------------

def get_interval_minutes(elapsed_hours: float,
                         last_growth_rate: Optional[float] = None) -> int:
    """
    Return the appropriate capture interval in minutes based on elapsed time
    and the most recently measured growth rate.

    Args:
        elapsed_hours:     Hours since the experiment started.
        last_growth_rate:  Most recent growth rate in mm/hr (any facet, max).
                           None if not yet measured.

    Returns:
        Interval in minutes until the next capture.
    """
    # Base interval from phase schedule
    if elapsed_hours < config.PHASE1_HOURS:
        interval = config.INTERVAL_PHASE1_MINS
    elif elapsed_hours < config.PHASE1_HOURS + config.PHASE2_HOURS:
        interval = config.INTERVAL_PHASE2_MINS
    else:
        interval = config.INTERVAL_PHASE3_MINS

    # Adaptive: tighten interval if growth is fast
    if last_growth_rate is not None:
        if last_growth_rate >= config.ADAPTIVE_FAST_THRESHOLD_MM_HR:
            interval = max(interval // 2, config.INTERVAL_MIN_MINS)
            logger.info(
                f"Fast growth detected ({last_growth_rate:.3f} mm/hr) — "
                f"interval tightened to {interval} min"
            )
        elif last_growth_rate < config.ADAPTIVE_SLOW_THRESHOLD_MM_HR:
            interval = min(interval * 2, config.INTERVAL_PHASE3_MINS)
            logger.info(
                f"Slow growth detected ({last_growth_rate:.3f} mm/hr) — "
                f"interval relaxed to {interval} min"
            )

    return interval


# ---------------------------------------------------------------------------
# Image cleanup
# ---------------------------------------------------------------------------

def keep_representative_image(image_folder: str,
                               run_id: str) -> Optional[str]:
    """
    Keep a single representative image from a capture run and delete the rest.

    The representative image is the one closest to 90° (side-on view),
    which gives the clearest silhouette of the crystal profile.

    Args:
        image_folder:  Path to the folder containing crystal_XXXX.jpg files.
        run_id:        Run identifier string (used in the saved filename).

    Returns:
        Path to the saved representative image, or None if no images found.
    """
    folder = Path(image_folder)
    images = sorted(folder.glob("crystal_*.jpg"))

    if not images:
        logger.warning(f"No images found in {image_folder} to clean up.")
        return None

    # Find the image closest to 90° (side-on)
    def angle_from_name(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 999

    target_angle = 90
    representative = min(images, key=lambda p: abs(angle_from_name(p) - target_angle))

    # Copy representative to the archive folder before deleting others
    archive_dir = Path(config.DATA_DIR) / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{run_id}_representative.jpg"
    shutil.copy2(representative, archive_path)
    logger.info(f"Representative image saved: {archive_path}")

    # Delete all images in the run folder
    deleted = 0
    for img in images:
        try:
            img.unlink()
            deleted += 1
        except OSError as e:
            logger.warning(f"Could not delete {img}: {e}")

    # Remove the now-empty folder
    try:
        folder.rmdir()
    except OSError:
        pass  # Folder not empty (e.g. other files present) — leave it

    logger.info(f"Cleaned up {deleted} images from {image_folder}")
    return str(archive_path)


# ---------------------------------------------------------------------------
# Run scheduler
# ---------------------------------------------------------------------------

class GrowthMonitorScheduler:
    """
    Manages the timing of repeated capture+reconstruct+analyse cycles.

    Usage:
        scheduler = GrowthMonitorScheduler(
            run_callback=my_run_function,
            experiment_name="crystal_KDP_001"
        )
        scheduler.start()
    """

    def __init__(self,
                 run_callback: Callable[[str, int], Optional[float]],
                 experiment_name: str,
                 simulate: bool = False):
        """
        Args:
            run_callback:      Function called for each capture+analyse cycle.
                               Signature: run_callback(run_id, run_number) -> max_growth_rate_mm_hr
                               Returns None if analysis failed or not yet available.
            experiment_name:   Name prefix for run folders and database entries.
            simulate:          If True, skip real hardware and use the simulator.
        """
        self.run_callback = run_callback
        self.experiment_name = experiment_name
        self.simulate = simulate

        self.start_time: Optional[datetime] = None
        self.run_number: int = 0
        self.last_growth_rate: Optional[float] = None
        self._stop_requested: bool = False

    def start(self):
        """
        Start the monitoring loop. Runs indefinitely until stop() is called
        or a KeyboardInterrupt is received.
        """
        self.start_time = datetime.now()
        self._stop_requested = False

        logger.info(
            f"Crystal growth monitoring started: {self.experiment_name}\n"
            f"  Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Simulate:   {self.simulate}"
        )

        try:
            while not self._stop_requested:
                self._execute_run()

                if self._stop_requested:
                    break

                elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
                interval = get_interval_minutes(elapsed, self.last_growth_rate)
                next_run = datetime.now() + timedelta(minutes=interval)

                logger.info(
                    f"Next run in {interval} min "
                    f"(at {next_run.strftime('%H:%M:%S')}). "
                    f"Elapsed: {elapsed:.1f} hr"
                )

                self._wait(interval * 60)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user (Ctrl+C).")

        logger.info(
            f"Experiment ended. Total runs: {self.run_number}. "
            f"Duration: {self._elapsed_str()}"
        )

    def stop(self):
        """Request a graceful stop after the current run completes."""
        self._stop_requested = True
        logger.info("Stop requested — will finish current run then exit.")

    def _execute_run(self):
        """Execute a single capture+reconstruct+analyse cycle."""
        self.run_number += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.experiment_name}_{timestamp}"

        logger.info(
            f"\n{'='*60}\n"
            f"Run {self.run_number} — {timestamp}\n"
            f"{'='*60}"
        )

        try:
            growth_rate = self.run_callback(run_id, self.run_number)
            if growth_rate is not None:
                self.last_growth_rate = growth_rate
        except Exception as e:
            logger.error(f"Run {self.run_number} failed: {e}", exc_info=True)
            # Continue to next run rather than crashing the whole experiment

    def _wait(self, seconds: int):
        """
        Wait for the specified number of seconds, checking for stop requests
        every 5 seconds so the loop can be interrupted cleanly.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self._stop_requested:
            time.sleep(min(5, deadline - time.monotonic()))

    def _elapsed_str(self) -> str:
        if self.start_time is None:
            return "0h 0m"
        delta = datetime.now() - self.start_time
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m = rem // 60
        return f"{h}h {m}m"

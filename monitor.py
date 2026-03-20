"""
monitor.py — Crystal growth monitoring entry point.

Runs a complete, automated crystal growth experiment:
  1. Captures 360° images on a schedule
  2. Reconstructs a 3D point cloud from each image set
  3. Detects crystal facets using RANSAC plane fitting
  4. Computes growth rates and stores them in a SQLite database
  5. Serves a live dashboard at http://localhost:5050
  6. Cleans up raw images after each run, keeping one representative image

Usage:
    # Start a new experiment
    python monitor.py --name crystal_KDP_001

    # Simulate (no hardware required)
    python monitor.py --name test_run --simulate

    # Resume an existing experiment (appends to existing database)
    python monitor.py --name crystal_KDP_001 --resume

    # Analyse existing data without capturing
    python monitor.py --name crystal_KDP_001 --analyse-only

    # Custom interval schedule
    python monitor.py --name crystal_KDP_001 --phase1-mins 10 --phase2-mins 20

    # Skip mesh generation (faster, point cloud only)
    python monitor.py --name crystal_KDP_001 --no-mesh --no-sam

Options:
    --name NAME           Experiment name (used for folder and database names)
    --simulate            Use simulated camera and stage (no hardware needed)
    --resume              Resume an existing experiment
    --analyse-only        Re-analyse existing data without capturing
    --no-sam              Use OpenCV thresholding instead of SAM segmentation
    --no-mesh             Skip mesh generation (point cloud only)
    --no-scale            Skip scale calibration
    --dashboard-port N    Dashboard port (default: 5050)
    --phase1-mins N       Interval for first 24h (default: from config)
    --phase2-mins N       Interval for days 2-7 (default: from config)
    --phase3-mins N       Interval after day 7 (default: from config)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup — file + console
# ---------------------------------------------------------------------------

def setup_logging(experiment_name: str) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{experiment_name}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ]
    )
    # Quieten noisy third-party loggers
    for noisy in ("werkzeug", "PIL", "matplotlib", "open3d"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("monitor")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crystal growth monitoring system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--name", required=True,
                        help="Experiment name (used for folders and database)")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated hardware (no camera or stage needed)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing experiment")
    parser.add_argument("--analyse-only", action="store_true",
                        help="Re-analyse existing data without capturing")
    parser.add_argument("--no-sam", action="store_true",
                        help="Use OpenCV thresholding instead of SAM")
    parser.add_argument("--no-mesh", action="store_true",
                        help="Skip mesh generation (point cloud only)")
    parser.add_argument("--no-scale", action="store_true",
                        help="Skip scale calibration")
    parser.add_argument("--dashboard-port", type=int, default=5050,
                        help="Port for the live dashboard (default: 5050)")
    parser.add_argument("--phase1-mins", type=int, default=None,
                        help="Capture interval for first 24h (minutes)")
    parser.add_argument("--phase2-mins", type=int, default=None,
                        help="Capture interval for days 2-7 (minutes)")
    parser.add_argument("--phase3-mins", type=int, default=None,
                        help="Capture interval after day 7 (minutes)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single run: capture → reconstruct → detect facets → store
# ---------------------------------------------------------------------------

def execute_run(run_id: str,
                run_number: int,
                experiment_name: str,
                args,
                tracker,
                db,
                logger: logging.Logger) -> float | None:
    """
    Execute one complete capture + reconstruct + analyse cycle.

    Returns the maximum growth rate in mm/hr, or None if not yet available.
    """
    from crystal_recon import config
    from crystal_recon.facets import detect_facets
    from crystal_recon.scheduler import keep_representative_image

    image_folder = str(Path(config.DATA_DIR) / run_id)
    timestamp = time.time()

    # ------------------------------------------------------------------
    # Step 1: Capture images
    # ------------------------------------------------------------------
    if not args.analyse_only:
        logger.info(f"[{run_id}] Step 1/4: Capturing images...")
        try:
            _capture_images(run_id, image_folder, args, logger)
        except Exception as e:
            logger.error(f"Capture failed: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Step 2: Reconstruct point cloud
    # ------------------------------------------------------------------
    logger.info(f"[{run_id}] Step 2/4: Reconstructing point cloud...")
    points = None
    try:
        points = _reconstruct(run_id, image_folder, args, logger)
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}", exc_info=True)

    if points is None or len(points) == 0:
        logger.error(f"[{run_id}] No points reconstructed — skipping analysis.")
        return None

    # ------------------------------------------------------------------
    # Step 3: Detect facets
    # ------------------------------------------------------------------
    logger.info(f"[{run_id}] Step 3/4: Detecting facets ({len(points)} points)...")
    try:
        from crystal_recon import config as cfg
        facets = detect_facets(
            points,
            max_facets=cfg.MAX_FACETS,
            min_inlier_fraction=cfg.MIN_FACET_INLIER_FRACTION,
            distance_threshold=cfg.FACET_DISTANCE_THRESHOLD,
            run_id=run_id,
            timestamp=timestamp,
        )
    except Exception as e:
        logger.error(f"Facet detection failed: {e}", exc_info=True)
        facets = []

    # ------------------------------------------------------------------
    # Step 4: Store results and clean up images
    # ------------------------------------------------------------------
    logger.info(f"[{run_id}] Step 4/4: Storing results and cleaning up...")

    # Record run in database
    db.add_run(run_id, timestamp, point_count=len(points))

    # Record facets
    for f in facets:
        db.add_facet(run_id, f.facet_id, f.normal, f.distance, f.area_pts)

    # Update tracker and compute growth rates
    tracker.update(facets, run_id, timestamp)

    # Store growth rates for any facets with at least 2 measurements
    for fid, ts_data in tracker.time_series.items():
        if len(ts_data.distances) >= 2:
            dt_hrs = (ts_data.timestamps[-1] - ts_data.timestamps[-2]) / 3600
            dd_mm = ts_data.distances[-1] - ts_data.distances[-2]
            rate = ts_data.growth_rate_mm_hr()
            if rate is not None and dt_hrs > 0:
                db.add_growth_rate(
                    fid,
                    ts_data.run_ids[-2],
                    ts_data.run_ids[-1],
                    dt_hrs, dd_mm, rate
                )

    # Keep one representative image, delete the rest
    rep_image = keep_representative_image(image_folder, run_id)
    if rep_image:
        # Update the run record with the image path
        db._conn.execute(
            "UPDATE runs SET representative_image = ? WHERE run_id = ?",
            (rep_image, run_id)
        )
        db._conn.commit()

    # Log summary
    logger.info(tracker.summary())
    max_rate = tracker.max_growth_rate()
    if max_rate is not None:
        logger.info(f"Max growth rate: {max_rate:.4f} mm/hr")

    return max_rate


def _capture_images(run_id: str, image_folder: str, args, logger):
    """Run the capture step (real hardware or simulator)."""
    from crystal_recon import config

    os.makedirs(image_folder, exist_ok=True)

    if args.simulate:
        # Use the simulator from capture.py
        import subprocess
        result = subprocess.run(
            [sys.executable, "capture.py",
             "--output", run_id,
             "--simulate",
             "--step", str(config.CAPTURE_STEP_DEGREES)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Capture simulator failed:\n{result.stderr}")
    else:
        # Real hardware capture
        import subprocess
        result = subprocess.run(
            [sys.executable, "capture.py",
             "--output", run_id,
             "--step", str(config.CAPTURE_STEP_DEGREES)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Capture failed:\n{result.stderr}")

    logger.info(f"Capture complete: {image_folder}")


def _reconstruct(run_id: str, image_folder: str, args, logger):
    """Run the reconstruction step and return the point cloud as numpy array."""
    import numpy as np
    from crystal_recon import config
    from crystal_recon.image_utils import validate_folder
    from crystal_recon.segmentation import (
        load_sam_predictor, make_mask_sam, make_mask_opencv, get_boundary
    )
    from crystal_recon.reconstruction import (
        contour_to_3d_with_depth, michelangelo, scale_coords
    )

    # Validate the image folder
    try:
        validate_folder(image_folder)
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(str(e))

    # Load SAM if needed
    predictor = None
    if not args.no_sam:
        sam_path = str(Path(config.DATA_DIR) / config.SAM_CHECKPOINT)
        if Path(sam_path).exists():
            try:
                predictor = load_sam_predictor(sam_path, config.SAM_MODEL_TYPE)
            except Exception as e:
                logger.warning(f"Could not load SAM ({e}) — falling back to OpenCV")
        else:
            logger.warning(f"SAM weights not found at {sam_path} — using OpenCV")

    # Collect images
    import cv2 as cv
    images = sorted(Path(image_folder).glob("crystal_*.jpg"))
    angle_step = config.CAPTURE_STEP_DEGREES
    all_contours = []
    angles = []

    for img_path in images:
        try:
            angle = int(img_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        img = cv.imread(str(img_path))
        if img is None:
            continue

        # Scale image
        h, w = img.shape[:2]
        new_w = int(w * config.IMAGE_SCALE_FACTOR)
        new_h = int(h * config.IMAGE_SCALE_FACTOR)
        img_small = cv.resize(img, (new_w, new_h))

        # Segment
        try:
            if predictor is not None:
                mask = make_mask_sam(img_small, predictor, config.SAM_BBOX)
            else:
                mask = make_mask_opencv(img_small)
            contour = get_boundary(mask)
            all_contours.append(contour)
            angles.append(angle)
        except Exception as e:
            logger.warning(f"Skipping {img_path.name}: {e}")
            continue

    if not all_contours:
        raise RuntimeError("No valid contours extracted from images")

    # Build 3D block and carve
    logger.info(f"  Carving {len(all_contours)} contours...")
    block = michelangelo(all_contours, angles, config.Z_AXIS_STEP_SIZE)

    if len(block) == 0:
        raise RuntimeError("Reconstruction produced empty point cloud")

    # Scale to mm
    if not args.no_scale:
        try:
            block = scale_coords(block, config.CALIBRATION_FILE,
                                 config.SCALE_CORRECTION)
        except Exception as e:
            logger.warning(f"Scale calibration failed ({e}) — using pixel units")

    return block


# ---------------------------------------------------------------------------
# Analyse-only mode: re-run facet detection on existing point clouds
# ---------------------------------------------------------------------------

def run_analyse_only(experiment_name: str, db, tracker, logger):
    """Re-detect facets from existing .asc point cloud files."""
    import numpy as np
    from crystal_recon import config
    from crystal_recon.facets import detect_facets

    output_dir = Path(config.OUTPUT_DIR)
    asc_files = sorted(output_dir.glob(f"{experiment_name}_*.asc"))

    if not asc_files:
        logger.error(f"No .asc files found in {output_dir} for {experiment_name}")
        return

    logger.info(f"Re-analysing {len(asc_files)} point cloud files...")

    for asc_path in asc_files:
        run_id = asc_path.stem
        timestamp = asc_path.stat().st_mtime

        try:
            points = np.loadtxt(str(asc_path))
        except Exception as e:
            logger.warning(f"Could not load {asc_path}: {e}")
            continue

        facets = detect_facets(
            points, run_id=run_id, timestamp=timestamp
        )

        db.add_run(run_id, timestamp, point_count=len(points))
        for f in facets:
            db.add_facet(run_id, f.facet_id, f.normal, f.distance, f.area_pts)
        tracker.update(facets, run_id, timestamp)

    logger.info(tracker.summary())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Apply any command-line interval overrides to config
    from crystal_recon import config
    if args.phase1_mins:
        config.INTERVAL_PHASE1_MINS = args.phase1_mins
    if args.phase2_mins:
        config.INTERVAL_PHASE2_MINS = args.phase2_mins
    if args.phase3_mins:
        config.INTERVAL_PHASE3_MINS = args.phase3_mins

    logger = setup_logging(args.name)
    logger.info(f"Crystal Growth Monitor starting — experiment: {args.name}")
    logger.info(f"  Simulate: {args.simulate}")
    logger.info(f"  SAM:      {not args.no_sam}")
    logger.info(f"  Mesh:     {not args.no_mesh}")
    logger.info(f"  Intervals: {config.INTERVAL_PHASE1_MINS}m / "
                f"{config.INTERVAL_PHASE2_MINS}m / {config.INTERVAL_PHASE3_MINS}m")

    # Check for existing experiment
    from crystal_recon.database import GrowthDatabase
    db_path = Path(config.OUTPUT_DIR) / f"{args.name}.db"
    if db_path.exists() and not args.resume and not args.analyse_only:
        logger.error(
            f"Database already exists: {db_path}\n"
            f"Use --resume to continue, or choose a different --name."
        )
        sys.exit(1)

    # Initialise database and tracker
    from crystal_recon.database import GrowthDatabase
    from crystal_recon.facets import FacetTracker
    db = GrowthDatabase(args.name)
    tracker = FacetTracker()

    # Start dashboard
    from crystal_recon.dashboard import Dashboard
    dashboard = Dashboard(db, args.name, port=args.dashboard_port)
    dashboard.start(start_time=time.time())
    logger.info(f"Dashboard: http://localhost:{args.dashboard_port}")

    # Analyse-only mode
    if args.analyse_only:
        run_analyse_only(args.name, db, tracker, logger)
        logger.info("Analysis complete.")
        db.close()
        return

    # Monitoring loop
    from crystal_recon.scheduler import GrowthMonitorScheduler

    def run_callback(run_id: str, run_number: int):
        return execute_run(
            run_id, run_number, args.name, args, tracker, db, logger
        )

    scheduler = GrowthMonitorScheduler(
        run_callback=run_callback,
        experiment_name=args.name,
        simulate=args.simulate,
    )

    # Wire dashboard countdown to scheduler
    _orig_wait = scheduler._wait
    def _wait_with_dashboard(seconds):
        dashboard.set_next_run_mins(int(seconds / 60))
        _orig_wait(seconds)
    scheduler._wait = _wait_with_dashboard

    try:
        scheduler.start()
    finally:
        db.close()
        logger.info("Experiment complete.")


if __name__ == "__main__":
    main()

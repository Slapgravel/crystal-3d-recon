"""
crystal_recon/interactive.py — Guided interactive prompt helpers.

Called by capture.py and monitor.py when they are run with no arguments.
Walks the user through all options with plain-English questions and returns
a populated argparse.Namespace so the rest of each script runs unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Low-level prompt utilities
# ---------------------------------------------------------------------------

def _ask(question: str, default: str | None = None) -> str:
    """Print a question and return the user's answer (stripped)."""
    if default is not None:
        prompt = f"  {question} [{default}]: "
    else:
        prompt = f"  {question}: "
    try:
        answer = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        sys.exit(0)
    return answer if answer else (default or "")


def _ask_yn(question: str, default: bool = False) -> bool:
    """Ask a yes/no question. Returns True for yes."""
    hint = "Y/n" if default else "y/N"
    raw = _ask(f"{question} ({hint})", default="y" if default else "n")
    return raw.lower() in ("y", "yes")


def _ask_int(question: str, default: int, min_val: int = 1,
             max_val: int = 10_000) -> int:
    """Ask for an integer, re-prompting on invalid input."""
    while True:
        raw = _ask(question, default=str(default))
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"    Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("    Please enter a whole number.")


def _choose(question: str, options: list[tuple[str, str]],
            default_index: int = 0) -> int:
    """
    Present a numbered menu and return the chosen index into `options`.

    `options` is a list of (key, description) tuples.
    """
    print(f"\n  {question}")
    for i, (key, desc) in enumerate(options):
        marker = " (default)" if i == default_index else ""
        print(f"    [{i + 1}] {key} — {desc}{marker}")
    while True:
        raw = _ask("Enter number", default=str(default_index + 1))
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"    Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("    Please enter a whole number.")


def _header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# capture.py interactive prompt
# ---------------------------------------------------------------------------

def capture_interactive() -> "argparse.Namespace":
    """
    Walk the user through all capture.py options interactively.
    Returns a Namespace compatible with capture.py's parse_args() output.
    """
    import argparse
    from crystal_recon import config

    _header("Crystal Image Capture — guided setup")
    print("  Press Enter to accept the default shown in [brackets].\n")

    # --- Mode ---
    mode_idx = _choose(
        "What would you like to do?",
        [
            ("Capture",          "Use the real camera and rotation stage"),
            ("Camera test",      "Use the real camera only (no stage connected yet)"),
            ("Simulate",         "Generate synthetic test images (no hardware needed)"),
            ("Calibrate",        "Capture checkerboard calibration images"),
            ("List hardware",    "Show connected cameras and stages, then exit"),
        ],
        default_index=0,
    )

    simulate   = mode_idx == 2
    calibrate  = mode_idx == 3
    no_stage   = mode_idx == 1
    list_cams  = mode_idx == 4

    if list_cams:
        # Return a minimal namespace — main() will call list_hardware() and exit
        return argparse.Namespace(
            output=None, simulate=False, calibrate=False,
            step=config.CAPTURE_STEP_DEGREES, camera=None,
            port=config.ZABER_PORT, cti=config.CAMERA_CTI_PATH,
            list_cameras=True, no_stage=False, no_home=False,
            output_dir="notebooks",
        )

    # --- Output name ---
    print()
    output = ""
    while not output:
        output = _ask("Output folder name (e.g. KDP_run1_ring_light)")
        if not output:
            print("    A folder name is required.")

    # --- Rotation step (not relevant for camera-test or simulate with default) ---
    step = config.CAPTURE_STEP_DEGREES
    if not calibrate:
        step = _ask_int(
            f"Rotation step in degrees (smaller = more images, slower)",
            default=config.CAPTURE_STEP_DEGREES,
            min_val=1, max_val=180,
        )
        n_images = 360 // step
        print(f"    → {n_images} images will be captured per run.")

    # --- Advanced options (only shown if user wants them) ---
    camera_index = None
    port         = config.ZABER_PORT
    cti_path     = config.CAMERA_CTI_PATH
    no_home      = False
    output_dir   = "notebooks"

    if not simulate:
        print()
        advanced = _ask_yn("Configure advanced options (camera index, port, CTI path)?",
                           default=False)
        if advanced:
            cam_raw = _ask("Camera index (leave blank to auto-select)", default="")
            if cam_raw:
                try:
                    camera_index = int(cam_raw)
                except ValueError:
                    print("    Invalid — auto-selecting camera.")

            if not no_stage:
                port_raw = _ask(
                    "Zaber stage serial port (e.g. COM3 or /dev/ttyUSB0, "
                    "blank to auto-discover)",
                    default=config.ZABER_PORT or "",
                )
                port = port_raw or config.ZABER_PORT

                no_home = _ask_yn(
                    "Skip homing the stage? (only if already at home position)",
                    default=False,
                )

            cti_raw = _ask(
                "Path to GenICam CTI file (blank to auto-find)",
                default=config.CAMERA_CTI_PATH or "",
            )
            cti_path = cti_raw or config.CAMERA_CTI_PATH

        output_dir_raw = _ask(
            "Parent directory for output folder",
            default="notebooks",
        )
        output_dir = output_dir_raw or "notebooks"

    # --- Summary ---
    print()
    print("  ── Summary ──────────────────────────────────────────")
    print(f"  Mode        : {'Simulate' if simulate else 'Camera test (no stage)' if no_stage else 'Calibrate' if calibrate else 'Full capture'}")
    print(f"  Output      : {output_dir}/{output}/")
    if not calibrate:
        print(f"  Step        : {step}° ({360 // step} images)")
    if not simulate:
        print(f"  Camera      : {'auto' if camera_index is None else camera_index}")
        if not no_stage:
            print(f"  Stage port  : {'auto' if not port else port}")
    print()

    go = _ask_yn("Start capture?", default=True)
    if not go:
        print("  Cancelled.")
        sys.exit(0)

    return argparse.Namespace(
        output=output,
        simulate=simulate,
        calibrate=calibrate,
        step=step,
        camera=camera_index,
        port=port,
        cti=cti_path,
        list_cameras=False,
        no_stage=no_stage,
        no_home=no_home,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# monitor.py interactive prompt
# ---------------------------------------------------------------------------

def monitor_interactive() -> "argparse.Namespace":
    """
    Walk the user through all monitor.py options interactively.
    Returns a Namespace compatible with monitor.py's parse_args() output.
    """
    import argparse
    from crystal_recon import config

    _header("Crystal Growth Monitor — guided setup")
    print("  Press Enter to accept the default shown in [brackets].\n")

    # --- Experiment name ---
    name = ""
    while not name:
        name = _ask("Experiment name (e.g. KDP_001_25C_sat)")
        if not name:
            print("    An experiment name is required.")

    # --- Mode ---
    mode_idx = _choose(
        "Run mode",
        [
            ("New experiment",   "Start fresh — will fail if name already exists"),
            ("Resume",           "Continue an existing experiment"),
            ("Analyse only",     "Re-analyse existing data without capturing"),
        ],
        default_index=0,
    )
    resume       = mode_idx == 1
    analyse_only = mode_idx == 2

    # --- Hardware ---
    print()
    simulate = _ask_yn("Use simulated hardware? (no camera or stage needed)",
                       default=False)

    # --- Analysis options ---
    print()
    print("  ── Analysis options ─────────────────────────────────")
    no_sam   = not _ask_yn("Use SAM segmentation? (better quality, slower)",
                           default=True)
    no_mesh  = not _ask_yn("Generate 3D mesh? (required for volume estimates)",
                           default=True)
    no_scale = not _ask_yn("Run scale calibration?", default=True)

    # --- Capture intervals ---
    print()
    print("  ── Capture schedule ─────────────────────────────────")
    print("  How often should images be captured?")
    print("  The monitor uses three phases: first 24 h, days 2–7, after day 7.")
    customise = _ask_yn("Customise capture intervals?", default=False)

    phase1_mins = phase2_mins = phase3_mins = None
    if customise:
        phase1_mins = _ask_int(
            "Phase 1 interval — first 24 h (minutes)",
            default=config.INTERVAL_PHASE1_MINS, min_val=1, max_val=1440,
        )
        phase2_mins = _ask_int(
            "Phase 2 interval — days 2–7 (minutes)",
            default=config.INTERVAL_PHASE2_MINS, min_val=1, max_val=1440,
        )
        phase3_mins = _ask_int(
            "Phase 3 interval — after day 7 (minutes)",
            default=config.INTERVAL_PHASE3_MINS, min_val=1, max_val=1440,
        )

    # --- Dashboard ---
    print()
    no_dashboard = not _ask_yn(
        "Start the web dashboard? (requires network access; use desktop viewer "
        "if on a corporate network)",
        default=False,
    )
    dashboard_port = 5050
    if not no_dashboard:
        dashboard_port = _ask_int("Dashboard port", default=5050,
                                  min_val=1024, max_val=65535)

    # --- Summary ---
    print()
    print("  ── Summary ──────────────────────────────────────────")
    print(f"  Experiment  : {name}")
    print(f"  Mode        : {'Analyse only' if analyse_only else 'Resume' if resume else 'New'}")
    print(f"  Hardware    : {'Simulated' if simulate else 'Real'}")
    print(f"  SAM         : {'yes' if not no_sam else 'no (OpenCV fallback)'}")
    print(f"  Mesh        : {'yes' if not no_mesh else 'no'}")
    print(f"  Scale cal.  : {'yes' if not no_scale else 'no'}")
    if customise:
        print(f"  Intervals   : {phase1_mins}m / {phase2_mins}m / {phase3_mins}m")
    else:
        print(f"  Intervals   : {config.INTERVAL_PHASE1_MINS}m / "
              f"{config.INTERVAL_PHASE2_MINS}m / "
              f"{config.INTERVAL_PHASE3_MINS}m (defaults)")
    print(f"  Dashboard   : {'disabled' if no_dashboard else f'http://localhost:{dashboard_port}'}")
    print()

    go = _ask_yn("Start monitoring?", default=True)
    if not go:
        print("  Cancelled.")
        sys.exit(0)

    return argparse.Namespace(
        name=name,
        simulate=simulate,
        resume=resume,
        analyse_only=analyse_only,
        no_sam=no_sam,
        no_mesh=no_mesh,
        no_scale=no_scale,
        no_dashboard=no_dashboard,
        dashboard_port=dashboard_port,
        phase1_mins=phase1_mins,
        phase2_mins=phase2_mins,
        phase3_mins=phase3_mins,
    )

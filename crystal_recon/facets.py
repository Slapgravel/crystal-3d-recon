"""
facets.py — Crystal facet detection and tracking across time steps.

Uses RANSAC plane fitting on the reconstructed point cloud to detect
individual crystal facets, then tracks them consistently across multiple
time steps by matching normal vector directions.

Each facet is characterised by:
  - A unit normal vector (direction perpendicular to the facet surface)
  - A signed distance from the crystal centroid to the facet plane
  - An area estimate (number of inlier points × voxel area)

Growth rate for a facet is the rate of change of its distance from the
centroid over time, measured in mm/hr.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Facet:
    """
    Represents a single crystal facet detected in one reconstruction.

    Attributes:
        facet_id:      Consistent label across time steps (e.g. "F0", "F1").
                       Assigned by matching normal vectors between runs.
        normal:        Unit normal vector pointing outward from the crystal.
        distance:      Signed distance from the crystal centroid to the facet
                       plane, in mm (after scale calibration) or pixels.
        area_pts:      Number of point cloud points belonging to this facet.
        inlier_pts:    (N, 3) array of the point cloud points on this facet.
        run_id:        Identifier of the reconstruction run this came from.
        timestamp:     Unix timestamp of the reconstruction run.
    """
    facet_id: str
    normal: np.ndarray          # shape (3,)
    distance: float
    area_pts: int
    inlier_pts: np.ndarray      # shape (N, 3)
    run_id: str = ""
    timestamp: float = 0.0

    def plane_equation(self) -> tuple:
        """Return (a, b, c, d) where ax + by + cz + d = 0."""
        a, b, c = self.normal
        d = -self.distance
        return (a, b, c, d)


@dataclass
class FacetTimeSeries:
    """
    Tracks a single facet across all time steps of an experiment.

    Attributes:
        facet_id:    Consistent label for this facet.
        reference_normal: Normal vector from the first detection (reference).
        timestamps:  List of Unix timestamps.
        distances:   List of distances from centroid (mm or px).
        run_ids:     List of run identifiers.
    """
    facet_id: str
    reference_normal: np.ndarray
    timestamps: list = field(default_factory=list)
    distances: list = field(default_factory=list)
    run_ids: list = field(default_factory=list)

    def add(self, facet: Facet):
        self.timestamps.append(facet.timestamp)
        self.distances.append(facet.distance)
        self.run_ids.append(facet.run_id)

    def growth_rate_mm_hr(self) -> Optional[float]:
        """
        Compute the mean growth rate over the last two measurements.

        Returns:
            Growth rate in mm/hr, or None if fewer than 2 measurements.
        """
        if len(self.distances) < 2:
            return None
        dt_hrs = (self.timestamps[-1] - self.timestamps[-2]) / 3600.0
        if dt_hrs <= 0:
            return None
        dd = self.distances[-1] - self.distances[-2]
        return dd / dt_hrs

    def growth_rate_series(self) -> tuple:
        """
        Compute growth rate at each time step.

        Returns:
            (times_hr, rates_mm_hr) — both as numpy arrays.
            times_hr is elapsed hours from the first measurement.
        """
        if len(self.distances) < 2:
            return np.array([]), np.array([])

        t = np.array(self.timestamps)
        d = np.array(self.distances)
        t0 = t[0]

        times = (t[1:] + t[:-1]) / 2 - t0  # midpoint times
        dt = np.diff(t) / 3600.0            # hours
        dd = np.diff(d)                      # mm
        rates = np.where(dt > 0, dd / dt, 0.0)

        return times / 3600.0, rates


# ---------------------------------------------------------------------------
# RANSAC plane fitting
# ---------------------------------------------------------------------------

def fit_plane_ransac(points: np.ndarray,
                     n_iterations: int = 1000,
                     distance_threshold: float = 1.5) -> tuple:
    """
    Fit a plane to a set of 3D points using RANSAC.

    Args:
        points:              (N, 3) array of 3D points.
        n_iterations:        Number of RANSAC iterations.
        distance_threshold:  Max distance (in same units as points) for a
                             point to be considered an inlier.

    Returns:
        (normal, d, inlier_mask) where:
          normal       — unit normal vector (3,)
          d            — plane offset (ax + by + cz = d)
          inlier_mask  — boolean array of shape (N,)
    """
    if len(points) < 3:
        raise ValueError(f"Need at least 3 points to fit a plane, got {len(points)}")

    best_inliers = np.zeros(len(points), dtype=bool)
    best_normal = np.array([0.0, 1.0, 0.0])
    best_d = 0.0
    best_count = 0

    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        # Sample 3 random points
        idx = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[idx]

        # Compute normal via cross product
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm

        # Plane equation: normal · x = d
        d = np.dot(normal, p0)

        # Count inliers
        distances = np.abs(points @ normal - d)
        inliers = distances < distance_threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal
            best_d = d

    # Refit using all inliers for a better estimate
    if best_count >= 3:
        inlier_pts = points[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(inlier_pts - centroid)
        best_normal = Vt[-1]
        best_d = np.dot(best_normal, centroid)

    return best_normal, best_d, best_inliers


# ---------------------------------------------------------------------------
# Multi-facet detection
# ---------------------------------------------------------------------------

def detect_facets(points: np.ndarray,
                  centroid: Optional[np.ndarray] = None,
                  max_facets: int = 12,
                  min_inlier_fraction: float = 0.03,
                  distance_threshold: float = 1.5,
                  run_id: str = "",
                  timestamp: float = 0.0) -> list:
    """
    Detect all major facets in a point cloud using iterative RANSAC.

    Repeatedly fits the dominant plane, removes its inliers, and fits the
    next plane until no more significant planes are found.

    Args:
        points:               (N, 3) array of 3D points (scaled, in mm).
        centroid:             Crystal centroid. If None, computed from points.
        max_facets:           Maximum number of facets to detect.
        min_inlier_fraction:  Minimum fraction of remaining points for a
                              plane to be considered a real facet.
        distance_threshold:   RANSAC inlier distance threshold (mm).
        run_id:               Run identifier for labelling.
        timestamp:            Unix timestamp for this run.

    Returns:
        List of Facet objects, sorted by area (largest first).
    """
    if centroid is None:
        centroid = points.mean(axis=0)

    remaining = points.copy()
    remaining_idx = np.arange(len(points))
    facets = []
    min_inliers = max(3, int(len(points) * min_inlier_fraction))

    for i in range(max_facets):
        if len(remaining) < min_inliers:
            break

        try:
            normal, d, inlier_mask = fit_plane_ransac(
                remaining, distance_threshold=distance_threshold
            )
        except ValueError:
            break

        n_inliers = np.sum(inlier_mask)
        if n_inliers < min_inliers:
            break

        # Orient normal to point outward from centroid
        centroid_d = np.dot(normal, centroid)
        if d < centroid_d:
            normal = -normal
            d = -d

        # Distance from centroid to facet plane
        dist_from_centroid = np.dot(normal, centroid) - d
        # Positive = centroid is on the inward side (correct orientation)
        # We want the outward distance: how far the facet is from the centre
        outward_distance = abs(np.dot(normal, remaining[inlier_mask].mean(axis=0)) - np.dot(normal, centroid))

        facet = Facet(
            facet_id=f"F{i}",
            normal=normal,
            distance=outward_distance,
            area_pts=n_inliers,
            inlier_pts=remaining[inlier_mask],
            run_id=run_id,
            timestamp=timestamp,
        )
        facets.append(facet)

        # Remove inliers from remaining points
        remaining = remaining[~inlier_mask]
        remaining_idx = remaining_idx[~inlier_mask]

        logger.debug(
            f"  Facet {i}: normal={normal.round(3)}, "
            f"dist={outward_distance:.2f}, inliers={n_inliers}"
        )

    logger.info(f"Detected {len(facets)} facets in {run_id} "
                f"({len(points)} points total)")
    return sorted(facets, key=lambda f: f.area_pts, reverse=True)


# ---------------------------------------------------------------------------
# Facet tracking across time steps
# ---------------------------------------------------------------------------

def match_facets(reference_facets: list,
                 new_facets: list,
                 angle_threshold_deg: float = 20.0) -> dict:
    """
    Match facets from a new run to the reference facets by normal direction.

    Uses the angle between normal vectors as the similarity metric. Each
    reference facet is matched to the closest new facet (greedy matching).

    Args:
        reference_facets:   List of Facet objects from the reference run.
        new_facets:         List of Facet objects from the new run.
        angle_threshold_deg: Maximum angle between normals to accept a match.

    Returns:
        Dict mapping reference facet_id -> matched Facet (or None if unmatched).
    """
    threshold_cos = np.cos(np.radians(angle_threshold_deg))
    matched = {}
    used = set()

    for ref in reference_facets:
        best_match = None
        best_cos = -1.0

        for i, new in enumerate(new_facets):
            if i in used:
                continue
            cos_sim = abs(np.dot(ref.normal, new.normal))
            if cos_sim > best_cos:
                best_cos = cos_sim
                best_match = (i, new)

        if best_match is not None and best_cos >= threshold_cos:
            idx, facet = best_match
            facet.facet_id = ref.facet_id  # Assign consistent ID
            matched[ref.facet_id] = facet
            used.add(idx)
            logger.debug(
                f"  Matched {ref.facet_id}: angle={np.degrees(np.arccos(best_cos)):.1f}°"
            )
        else:
            matched[ref.facet_id] = None
            logger.warning(
                f"  No match found for {ref.facet_id} "
                f"(best angle: {np.degrees(np.arccos(max(best_cos, 0))):.1f}°)"
            )

    return matched


class FacetTracker:
    """
    Maintains a consistent set of facet time series across all runs.

    Usage:
        tracker = FacetTracker()
        for run in runs:
            facets = detect_facets(run.point_cloud, ...)
            tracker.update(facets, run_id, timestamp)
        growth_rates = tracker.max_growth_rate()
    """

    def __init__(self, angle_threshold_deg: float = 20.0):
        self.angle_threshold_deg = angle_threshold_deg
        self.reference_facets: list = []
        self.time_series: dict = {}  # facet_id -> FacetTimeSeries
        self.run_count: int = 0

    def update(self, facets: list, run_id: str, timestamp: float):
        """
        Update the tracker with facets from a new run.

        On the first call, establishes the reference set. On subsequent
        calls, matches new facets to the reference and updates time series.

        Args:
            facets:     List of Facet objects from detect_facets().
            run_id:     Run identifier string.
            timestamp:  Unix timestamp of this run.
        """
        self.run_count += 1

        if not self.reference_facets:
            # First run — establish reference
            self.reference_facets = facets
            for f in facets:
                self.time_series[f.facet_id] = FacetTimeSeries(
                    facet_id=f.facet_id,
                    reference_normal=f.normal.copy(),
                )
                f.run_id = run_id
                f.timestamp = timestamp
                self.time_series[f.facet_id].add(f)
            logger.info(
                f"Reference established with {len(facets)} facets from {run_id}"
            )
            return

        # Subsequent runs — match to reference
        matched = match_facets(
            self.reference_facets, facets, self.angle_threshold_deg
        )

        for facet_id, facet in matched.items():
            if facet is None:
                logger.warning(f"Facet {facet_id} not detected in {run_id}")
                continue
            facet.run_id = run_id
            facet.timestamp = timestamp
            self.time_series[facet_id].add(facet)

        logger.info(
            f"Run {self.run_count}: matched "
            f"{sum(v is not None for v in matched.values())}/{len(self.reference_facets)} facets"
        )

    def max_growth_rate(self) -> Optional[float]:
        """
        Return the maximum growth rate across all tracked facets (mm/hr).
        Used by the scheduler to adapt the capture interval.
        """
        rates = [
            ts.growth_rate_mm_hr()
            for ts in self.time_series.values()
            if ts.growth_rate_mm_hr() is not None
        ]
        return max(rates) if rates else None

    def summary(self) -> str:
        """Return a human-readable summary of current growth rates."""
        lines = [f"Facet growth rates (run {self.run_count}):"]
        for fid, ts in sorted(self.time_series.items()):
            rate = ts.growth_rate_mm_hr()
            n = len(ts.distances)
            if rate is not None:
                lines.append(f"  {fid}: {rate:+.4f} mm/hr  ({n} measurements)")
            else:
                lines.append(f"  {fid}: -- mm/hr  ({n} measurement{'s' if n != 1 else ''})")
        return "\n".join(lines)

"""
Feature extraction from segmented letter images.

For each letter ROI we compute a fixed-length feature vector that captures
stroke-level properties relevant to handwriting style classification:

  1. Slant angle          – dominant angle of near-vertical strokes
  2. Stroke width mean    – average skeleton-distance-transform width
  3. Stroke width std     – variation in width (pressure sensitivity)
  4. Curvature mean       – average contour curvature (loopiness)
  5. Curvature std        – variation in curvature
  6. Aspect ratio         – bounding-box width / height
  7. Density              – ink-pixel ratio inside bounding box
  8. Loop count           – number of enclosed regions (holes)
  9. Connectivity         – ratio of ink pixels in the leftmost/rightmost
                            columns (how much the letter connects to neighbours)
  10. Flourish score      – extent of ink outside the core bounding box
                            (decorative strokes)
  11. Stroke length ratio – total skeleton length / diagonal of bbox
  12. Horizontal extent   – fraction of columns that contain ink
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from .preprocessor import LetterROI


NUM_FEATURES = 12


@dataclass
class LetterFeatures:
    """Computed feature vector for a single letter."""

    roi: LetterROI
    vector: np.ndarray  # shape (NUM_FEATURES,), float32, each in [0, 1]

    # Named access
    slant_angle: float = 0.0
    stroke_width_mean: float = 0.0
    stroke_width_std: float = 0.0
    curvature_mean: float = 0.0
    curvature_std: float = 0.0
    aspect_ratio: float = 0.0
    density: float = 0.0
    loop_count: float = 0.0
    connectivity: float = 0.0
    flourish_score: float = 0.0
    stroke_length_ratio: float = 0.0
    horizontal_extent: float = 0.0


def _safe_norm(value: float, lo: float, hi: float) -> float:
    """Clamp and normalise value to [0, 1]."""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _compute_slant(binary: np.ndarray) -> float:
    """Estimate dominant slant angle in degrees using Hough lines."""
    lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 180, threshold=10,
        minLineLength=max(5, binary.shape[0] // 4), maxLineGap=3,
    )
    if lines is None or len(lines) == 0:
        return 90.0  # vertical = no slant

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1) + 1e-6))
        if angle > 30:  # keep near-vertical strokes only
            angles.append(angle)

    return float(np.median(angles)) if angles else 90.0


def _compute_skeleton(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning to obtain 1-pixel-wide skeleton."""
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(binary)
    temp = binary.copy()
    while True:
        eroded = cv2.erode(temp, elem)
        dilated = cv2.dilate(eroded, elem)
        diff = cv2.subtract(temp, dilated)
        skel = cv2.bitwise_or(skel, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel


def _loop_count(binary: np.ndarray) -> int:
    """Count enclosed holes inside the letter."""
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape
    count = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Exclude the outer background contour
        if x > 0 and y > 0 and x + cw < w and y + ch < h:
            if cv2.contourArea(cnt) > 8:
                count += 1
    return count


def _contour_curvature(contour: np.ndarray) -> tuple[float, float]:
    """Mean and std of local curvature along the contour."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 5:
        return 0.0, 0.0

    curvatures = []
    for i in range(1, n - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        v1 = p0 - p1
        v2 = p2 - p1
        cross = float(np.cross(v1, v2))
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
        curv = abs(cross / denom)
        curvatures.append(curv)

    arr = np.array(curvatures)
    return float(np.mean(arr)), float(np.std(arr))


def extract_features(roi: LetterROI) -> LetterFeatures:
    """Compute the full feature vector for a letter ROI."""
    img = roi.image
    if img.size == 0 or img.shape[0] < 3 or img.shape[1] < 3:
        return LetterFeatures(roi=roi, vector=np.zeros(NUM_FEATURES, dtype=np.float32))

    h, w = img.shape[:2]
    diag = np.sqrt(h**2 + w**2)

    # --- 1. Slant ---
    raw_slant = _compute_slant(img)
    slant = _safe_norm(raw_slant, 45.0, 90.0)  # 45° heavy slant -> 0, 90° upright -> 1

    # --- 2-3. Stroke width (via distance transform on skeleton) ---
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    skel = _compute_skeleton(img)
    skel_mask = skel > 0
    if np.any(skel_mask):
        widths = dist[skel_mask]
        sw_mean = float(np.mean(widths))
        sw_std = float(np.std(widths))
    else:
        sw_mean, sw_std = 1.0, 0.0
    stroke_width_mean = _safe_norm(sw_mean, 0.5, 8.0)
    stroke_width_std = _safe_norm(sw_std, 0.0, 4.0)

    # --- 4-5. Curvature ---
    curv_mean, curv_std = _contour_curvature(roi.contour)
    curvature_mean = _safe_norm(curv_mean, 0.0, 1.0)
    curvature_std = _safe_norm(curv_std, 0.0, 0.5)

    # --- 6. Aspect ratio ---
    aspect_ratio = _safe_norm(w / (h + 1e-6), 0.2, 2.0)

    # --- 7. Density ---
    ink_pixels = cv2.countNonZero(img)
    density = ink_pixels / (h * w + 1e-6)
    density = _safe_norm(density, 0.0, 0.7)

    # --- 8. Loop count ---
    loops = _loop_count(img)
    loop_count = _safe_norm(float(loops), 0.0, 4.0)

    # --- 9. Connectivity ---
    left_col = img[:, :max(1, w // 8)]
    right_col = img[:, -(max(1, w // 8)):]
    left_ink = cv2.countNonZero(left_col) / (left_col.size + 1e-6)
    right_ink = cv2.countNonZero(right_col) / (right_col.size + 1e-6)
    connectivity = _safe_norm((left_ink + right_ink) / 2, 0.0, 0.5)

    # --- 10. Flourish score ---
    # Measure ink that lies outside the tight convex hull
    hull = cv2.convexHull(roi.contour)
    hull_mask = np.zeros_like(img)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    outside_ink = cv2.countNonZero(cv2.bitwise_and(img, cv2.bitwise_not(hull_mask)))
    flourish_score = _safe_norm(outside_ink / (ink_pixels + 1e-6), 0.0, 0.3)

    # --- 11. Stroke length ratio ---
    skel_len = float(np.sum(skel_mask))
    stroke_length_ratio = _safe_norm(skel_len / (diag + 1e-6), 0.5, 5.0)

    # --- 12. Horizontal extent ---
    col_sums = np.sum(img, axis=0)
    active_cols = np.sum(col_sums > 0)
    horizontal_extent = _safe_norm(active_cols / (w + 1e-6), 0.3, 1.0)

    vec = np.array(
        [
            slant, stroke_width_mean, stroke_width_std,
            curvature_mean, curvature_std, aspect_ratio,
            density, loop_count, connectivity, flourish_score,
            stroke_length_ratio, horizontal_extent,
        ],
        dtype=np.float32,
    )

    return LetterFeatures(
        roi=roi,
        vector=vec,
        slant_angle=slant,
        stroke_width_mean=stroke_width_mean,
        stroke_width_std=stroke_width_std,
        curvature_mean=curvature_mean,
        curvature_std=curvature_std,
        aspect_ratio=aspect_ratio,
        density=density,
        loop_count=loop_count,
        connectivity=connectivity,
        flourish_score=flourish_score,
        stroke_length_ratio=stroke_length_ratio,
        horizontal_extent=horizontal_extent,
    )

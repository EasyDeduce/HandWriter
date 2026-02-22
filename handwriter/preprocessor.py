"""
Image preprocessing and letter segmentation for handwriting analysis.

Pipeline:
  1. Load image (grayscale or blue-ink-optimised channel extraction)
  2. Adaptive binarisation (Sauvola thresholding)
  3. Noise removal (morphological opening)
  4. Line segmentation via horizontal projection profile
  5. Letter segmentation via vertical projection + contour refinement
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class LetterROI:
    """Region-of-interest for a single segmented letter."""

    image: np.ndarray  # Binary crop of the letter (white ink on black bg)
    contour: np.ndarray  # Enclosing contour points
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in the original image
    char_index: int  # Position index within the line
    line_index: int  # Which text line this belongs to
    is_punctuation: bool = False  # True for segments too small to be a letter


@dataclass
class PreprocessResult:
    """Full result of the preprocessing pipeline."""

    original: np.ndarray
    gray: np.ndarray
    binary: np.ndarray
    lines: list[np.ndarray]  # Binary image for each text line
    letters: list[LetterROI] = field(default_factory=list)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk. Raises FileNotFoundError if missing."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at: {path}")
    return img


def to_grayscale(img: np.ndarray, ink_color: str = "dark") -> np.ndarray:
    """Convert to grayscale optimised for the ink colour.

    Parameters
    ----------
    ink_color : str
        "dark"  – standard dark ink on light paper (default).
        "blue"  – deep-blue ink on white paper.  Extracts the channel
                  with best ink/paper contrast, normalises the dynamic
                  range, and applies CLAHE to handle uneven lighting
                  from phone-camera photos.
    """
    if len(img.shape) == 2:
        return img

    if ink_color == "blue":
        # --- Channel selection -----------------------------------------
        # Blue ink absorbs red light, so the red channel has the best
        # ink-vs-paper contrast.  We also compute (R – B) which further
        # isolates the blue dye.  Whichever has wider spread wins.
        b_ch = img[:, :, 0].astype(np.float32)
        r_ch = img[:, :, 2].astype(np.float32)

        # (R − B): paper ≈ 0, blue ink → negative  →  invert
        diff = r_ch - b_ch
        diff_range = diff.max() - diff.min()
        r_range = float(r_ch.max() - r_ch.min())

        if diff_range > r_range * 0.6:
            # R−B channel has meaningful contrast → use it
            gray = ((diff - diff.min()) / (diff_range + 1e-6) * 255).astype(np.uint8)
        else:
            # Fall back to raw red channel (still best single channel)
            gray = r_ch.astype(np.uint8)

        # --- Normalise to full 0-255 range -----------------------------
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore[call-overload]

        # --- CLAHE (contrast-limited adaptive histogram equalisation) ---
        # Handles shadows and uneven lighting typical of phone photos.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # --- Light Gaussian blur to reduce JPEG artefacts --------------
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        return gray

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(
    gray: np.ndarray,
    block_size: int = 25,
    C: int = 12,
    ink_color: str = "dark",
) -> np.ndarray:
    """Binarise a grayscale image so that ink pixels are white (255).

    For blue ink the image has already been contrast-enhanced, so we
    use Otsu's global threshold combined with a local adaptive pass
    for best results on uneven backgrounds.
    """
    if ink_color == "blue":
        # With CLAHE-enhanced grayscale, adaptive threshold alone works
        # well.  Use a slightly lower C to preserve thin strokes that
        # CLAHE has now made visible.
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, max(C - 4, 4),
        )
        return adaptive

    # Standard dark-ink path
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C
    )
    return binary


def denoise(binary: np.ndarray, kernel_size: int = 2, ink_color: str = "dark") -> np.ndarray:
    """Remove salt-and-pepper noise with morphological opening."""
    if ink_color == "blue":
        # 1) Morphological close to bridge tiny gaps in strokes,
        #    then open to remove isolated specks.
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def segment_lines(binary: np.ndarray, min_gap: int = 8) -> list[tuple[int, int]]:
    """
    Segment text lines using horizontal projection profile.

    Returns list of (y_start, y_end) row ranges.
    """
    h_proj = np.sum(binary, axis=1)
    threshold = h_proj.max() * 0.02

    in_line = False
    lines: list[tuple[int, int]] = []
    start = 0

    for y, val in enumerate(h_proj):
        if not in_line and val > threshold:
            in_line = True
            start = y
        elif in_line and val <= threshold:
            if y - start > min_gap:
                lines.append((start, y))
            in_line = False

    if in_line:
        lines.append((start, len(h_proj)))

    return lines


def segment_letters_in_line(
    line_img: np.ndarray,
    min_area: int = 30,
    merge_distance: int = 3,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]]:
    """
    Segment individual letters from a single text-line image.

    Uses contour detection with bounding-box merging for broken strokes.

    Returns list of (letter_crop, contour, (x, y, w, h)).
    """
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    contour_map: dict[int, np.ndarray] = {}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
        contour_map[len(boxes) - 1] = cnt

    if not boxes:
        return []

    # Sort left-to-right
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    boxes = [boxes[i] for i in order]
    new_contour_map = {j: contour_map[order[j]] for j in range(len(order))}
    contour_map = new_contour_map

    # Merge overlapping / very close bounding boxes
    merged: list[tuple[int, int, int, int]] = [boxes[0]]
    merged_contours: list[np.ndarray] = [contour_map[0]]

    for i in range(1, len(boxes)):
        px, py, pw, ph = merged[-1]
        cx, cy, cw, ch = boxes[i]
        if cx <= px + pw + merge_distance:
            nx = min(px, cx)
            ny = min(py, cy)
            nw = max(px + pw, cx + cw) - nx
            nh = max(py + ph, cy + ch) - ny
            merged[-1] = (nx, ny, nw, nh)
            merged_contours[-1] = np.vstack([merged_contours[-1], contour_map[i]])
        else:
            merged.append(boxes[i])
            merged_contours.append(contour_map[i])

    results = []
    for (x, y, w, h), cnt in zip(merged, merged_contours):
        pad = 2
        y0 = max(y - pad, 0)
        y1 = min(y + h + pad, line_img.shape[0])
        x0 = max(x - pad, 0)
        x1 = min(x + w + pad, line_img.shape[1])
        crop = line_img[y0:y1, x0:x1].copy()
        results.append((crop, cnt, (x, y, w, h)))

    return results


def preprocess(image_path: str, ink_color: str = "dark") -> PreprocessResult:
    """Run the full preprocessing pipeline on a handwriting image.

    Parameters
    ----------
    ink_color : str
        ``"dark"`` for standard dark ink, ``"blue"`` for deep-blue ink
        on white paper.
    """
    original = load_image(image_path)
    gray = to_grayscale(original, ink_color=ink_color)
    binary = binarize(gray, ink_color=ink_color)
    binary = denoise(binary, ink_color=ink_color)

    line_ranges = segment_lines(binary)
    lines: list[np.ndarray] = []
    all_letters: list[LetterROI] = []

    for line_idx, (y_start, y_end) in enumerate(line_ranges):
        line_img = binary[y_start:y_end, :]
        lines.append(line_img)

        line_height = y_end - y_start

        letter_segments = segment_letters_in_line(line_img)
        for char_idx, (crop, contour, bbox) in enumerate(letter_segments):
            # Adjust bbox y-coordinates to original image space
            global_bbox = (bbox[0], bbox[1] + y_start, bbox[2], bbox[3])

            # Mark very small segments as punctuation so the classifier
            # can treat them differently.  Heuristic: both width and
            # height must be < 25 % of the line height.
            _x, _y, bw, bh = bbox
            is_punct = (bw < line_height * 0.25) and (bh < line_height * 0.25)

            roi = LetterROI(
                image=crop,
                contour=contour,
                bbox=global_bbox,
                char_index=char_idx,
                line_index=line_idx,
                is_punctuation=is_punct,
            )
            all_letters.append(roi)

    return PreprocessResult(
        original=original,
        gray=gray,
        binary=binary,
        lines=lines,
        letters=all_letters,
    )

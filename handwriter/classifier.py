"""
Handwriting style classifier.

Scores each segmented letter against six canonical handwriting styles
using a feature-profile matching approach.

Styles
------
1. Print          – Disconnected, uniform width, upright, simple geometric forms.
2. D'Nealian      – Slightly slanted, rounded, moderate connections, tail hooks.
3. Cursive        – Fully connected, flowing, looped, moderate-to-heavy slant.
4. Italic Neat    – Consistent right slant, angular, clean, slight connections.
5. Spencerian     – Ornamental, heavy pressure variation, extreme slant, long strokes.
6. Calligraphy    – Dramatic thick-thin contrast, decorative flourishes, artistic.

Each style is represented as an *ideal feature profile* (length-12 vector)
matching the features from `feature_extractor.py`, plus a weight vector that
controls which features matter most for each style.

The per-letter score for a style is computed as the weighted cosine similarity
between the letter's feature vector and the style's ideal profile, clamped to
[0, 1].
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .feature_extractor import LetterFeatures, NUM_FEATURES


# Feature order (must match feature_extractor.extract_features):
# 0  slant_angle         (0 = heavy slant, 1 = upright)
# 1  stroke_width_mean
# 2  stroke_width_std    (pressure variation)
# 3  curvature_mean
# 4  curvature_std
# 5  aspect_ratio
# 6  density
# 7  loop_count
# 8  connectivity        (edge ink = connects to neighbours)
# 9  flourish_score
# 10 stroke_length_ratio
# 11 horizontal_extent


STYLE_NAMES: list[str] = [
    "Print",
    "D'Nealian",
    "Cursive",
    "Italic Neat",
    "Spencerian",
    "Calligraphy",
]

# Ideal feature profiles for each style (12-D, values in [0,1]).
# These are hand-tuned based on typographic research.
_IDEAL_PROFILES = np.array(
    [
        # Print: upright, uniform width, low curvature, no loops, disconnected
        [1.0, 0.4, 0.1, 0.25, 0.15, 0.5, 0.45, 0.2, 0.05, 0.0, 0.4, 0.7],
        # D'Nealian: slight slant, rounded, moderate connectivity, medium loops
        [0.75, 0.45, 0.15, 0.50, 0.25, 0.55, 0.50, 0.35, 0.35, 0.05, 0.5, 0.75],
        # Cursive: slanted, connected, loopy, flowing
        [0.40, 0.45, 0.25, 0.65, 0.35, 0.60, 0.55, 0.60, 0.80, 0.10, 0.70, 0.90],
        # Italic Neat: right-slant, angular, clean, slight connections
        [0.35, 0.40, 0.10, 0.35, 0.15, 0.45, 0.40, 0.15, 0.25, 0.02, 0.50, 0.70],
        # Spencerian: extreme slant, pressure variation, ornamental, long strokes
        [0.15, 0.50, 0.55, 0.70, 0.50, 0.50, 0.45, 0.55, 0.75, 0.30, 0.85, 0.85],
        # Calligraphy: thick-thin contrast, flourishes, decorative
        [0.55, 0.65, 0.70, 0.55, 0.45, 0.55, 0.55, 0.40, 0.30, 0.60, 0.65, 0.80],
    ],
    dtype=np.float32,
)

# Per-style feature importance weights (higher = this feature matters more).
_STYLE_WEIGHTS = np.array(
    [
        # Print: slant, uniformity, disconnection matter most
        [2.5, 1.0, 2.0, 1.0, 0.8, 0.8, 0.6, 1.2, 2.5, 1.5, 0.5, 0.6],
        # D'Nealian: curvature, connectivity, loops
        [1.5, 0.8, 1.0, 2.0, 1.2, 0.8, 0.6, 1.5, 2.0, 0.8, 0.8, 0.6],
        # Cursive: connectivity, loops, slant
        [2.0, 0.6, 1.0, 1.5, 1.0, 0.6, 0.5, 2.5, 3.0, 0.8, 1.5, 1.2],
        # Italic Neat: slant, angularity (low curvature), cleanness (low width std)
        [2.5, 1.0, 2.0, 2.0, 1.5, 0.8, 0.6, 1.0, 1.5, 1.0, 0.8, 0.6],
        # Spencerian: slant, pressure, flourish, stroke length
        [2.0, 1.5, 2.5, 1.2, 1.2, 0.6, 0.5, 1.5, 1.5, 2.5, 2.0, 0.8],
        # Calligraphy: width variation, flourish dominate
        [1.0, 2.0, 3.0, 1.0, 1.0, 0.6, 0.5, 1.0, 0.8, 3.0, 1.2, 0.8],
    ],
    dtype=np.float32,
)


@dataclass
class StyleScores:
    """Per-letter classification result."""

    letter_index: int
    line_index: int
    scores: dict[str, float]  # style_name -> score in [0, 1]
    is_punctuation: bool = False  # True for segments too small to be a letter

    @property
    def best_style(self) -> str:
        return max(self.scores, key=self.scores.get)  # type: ignore[arg-type]

    def __str__(self) -> str:
        parts = [f"{name}: {score:.3f}" for name, score in self.scores.items()]
        return f"Letter(line={self.line_index}, idx={self.letter_index}) -> {' | '.join(parts)}"


def _weighted_cosine(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted cosine similarity, result in [0, 1]."""
    wa = a * w
    wb = b * w
    dot = np.dot(wa, wb)
    norm_a = np.linalg.norm(wa) + 1e-8
    norm_b = np.linalg.norm(wb) + 1e-8
    sim = dot / (norm_a * norm_b)
    return float(np.clip(sim, 0.0, 1.0))


def _weighted_distance(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted Euclidean distance (lower = closer match)."""
    diff = (a - b) * w
    return float(np.sqrt(np.dot(diff, diff)))


def classify_letter(features: LetterFeatures) -> StyleScores:
    """Score a single letter against all six handwriting styles.
    
    Uses weighted Euclidean distance converted to a probability
    distribution via softmax so that (a) the best style is clearly
    highlighted and (b) scores across styles sum to 1.
    """
    vec = features.vector

    # Compute raw distances (lower = better match)
    dists = np.array([
        _weighted_distance(vec, _IDEAL_PROFILES[i], _STYLE_WEIGHTS[i])
        for i in range(len(STYLE_NAMES))
    ])

    # Convert to similarity: exp(-temperature * dist)
    # Temperature controls how "peaky" the distribution is.
    temperature = 2.5
    neg_dists = -temperature * dists
    neg_dists -= neg_dists.max()          # numerical stability
    exp_vals = np.exp(neg_dists)
    probs = exp_vals / (exp_vals.sum() + 1e-8)

    scores: dict[str, float] = {}
    for i, name in enumerate(STYLE_NAMES):
        scores[name] = round(float(probs[i]), 4)

    return StyleScores(
        letter_index=features.roi.char_index,
        line_index=features.roi.line_index,
        scores=scores,
    )


def classify_all(features_list: list[LetterFeatures]) -> list[StyleScores]:
    """Classify every letter in the list."""
    return [classify_letter(f) for f in features_list]

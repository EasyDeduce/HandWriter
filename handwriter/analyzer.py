"""
High-level analysis pipeline.

Orchestrates: preprocess -> feature extraction -> classification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .preprocessor import preprocess, PreprocessResult, LetterROI
from .feature_extractor import extract_features, LetterFeatures
from .classifier import classify_all, StyleScores, STYLE_NAMES


@dataclass
class CombinationScores:
    """Style scores for a consecutive letter pair (bigram)."""

    line_index: int
    first_index: int
    second_index: int
    label: str  # e.g. "L0+L1"
    scores: dict[str, float]  # averaged style scores

    @property
    def best_style(self) -> str:
        return max(self.scores, key=self.scores.get)  # type: ignore[arg-type]


@dataclass
class AnalysisResult:
    """Complete analysis result for a handwriting image."""

    image_path: str
    num_lines: int
    num_letters: int
    letter_scores: list[StyleScores]
    combination_scores: list[CombinationScores] = field(default_factory=list)
    aggregate_scores: dict[str, float] = field(default_factory=dict)

    def compute_aggregate(self) -> None:
        """Compute mean style score across all *non-punctuation* letters."""
        real = [ls for ls in self.letter_scores if not ls.is_punctuation]
        if not real:
            self.aggregate_scores = {name: 0.0 for name in STYLE_NAMES}
            return

        totals = {name: 0.0 for name in STYLE_NAMES}
        for ls in real:
            for name in STYLE_NAMES:
                totals[name] += ls.scores[name]

        n = len(real)
        self.aggregate_scores = {name: round(totals[name] / n, 4) for name in STYLE_NAMES}

    def compute_combination_scores(self) -> None:
        """Compute averaged style scores for every consecutive letter pair per line."""
        from collections import defaultdict

        # Group letter scores by line
        by_line: dict[int, list[StyleScores]] = defaultdict(list)
        for ls in self.letter_scores:
            by_line[ls.line_index].append(ls)

        combos: list[CombinationScores] = []
        for line_idx in sorted(by_line):
            line_letters = sorted(by_line[line_idx], key=lambda s: s.letter_index)
            # Only use non-punctuation letters for combination scoring
            real_letters = [l for l in line_letters if not l.is_punctuation]
            for i in range(len(real_letters) - 1):
                a = real_letters[i]
                b = real_letters[i + 1]
                avg_scores = {
                    name: round((a.scores[name] + b.scores[name]) / 2, 4)
                    for name in STYLE_NAMES
                }
                combos.append(
                    CombinationScores(
                        line_index=line_idx,
                        first_index=a.letter_index,
                        second_index=b.letter_index,
                        label=f"L{a.letter_index}+L{b.letter_index}",
                        scores=avg_scores,
                    )
                )
        self.combination_scores = combos

    def to_dict(self) -> dict:
        """Serialisable dictionary."""
        return {
            "image_path": self.image_path,
            "num_lines": self.num_lines,
            "num_letters": self.num_letters,
            "aggregate_scores": self.aggregate_scores,
            "letter_scores": [
                {
                    "line": ls.line_index,
                    "index": ls.letter_index,
                    "scores": ls.scores,
                    "best_match": ls.best_style,
                }
                for ls in self.letter_scores
            ],
            "combination_scores": [
                {
                    "line": cs.line_index,
                    "pair": cs.label,
                    "scores": cs.scores,
                    "best_match": cs.best_style,
                }
                for cs in self.combination_scores
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def analyze(image_path: str, verbose: bool = False, ink_color: str = "dark") -> AnalysisResult:
    """
    Full analysis pipeline.

    Parameters
    ----------
    image_path : str
        Path to a handwriting image.
    verbose : bool
        If True, print progress to stdout.
    ink_color : str
        ``"dark"`` for standard dark ink, ``"blue"`` for deep-blue ink
        on white paper.
    """
    if verbose:
        print(f"[1/3] Preprocessing image: {image_path}")

    prep: PreprocessResult = preprocess(image_path, ink_color=ink_color)

    if verbose:
        print(f"      Found {len(prep.lines)} text line(s), {len(prep.letters)} letter(s)")
        print("[2/3] Extracting features ...")

    features: list[LetterFeatures] = [extract_features(roi) for roi in prep.letters]

    if verbose:
        print("[3/3] Classifying letters ...")

    scores: list[StyleScores] = classify_all(features)

    # Propagate punctuation flags from ROIs to scores
    for sc, roi in zip(scores, prep.letters):
        sc.is_punctuation = roi.is_punctuation

    result = AnalysisResult(
        image_path=image_path,
        num_lines=len(prep.lines),
        num_letters=len(prep.letters),
        letter_scores=scores,
    )
    result.compute_aggregate()
    result.compute_combination_scores()

    return result

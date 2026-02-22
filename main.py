#!/usr/bin/env python3
"""
HandWriter CLI — Handwriting style analyser.

Usage:
    python main.py                        # interactive mode (prompts for image path)
    python main.py --image path/to/img    # direct mode
    python main.py --image img --json     # output raw JSON
"""

from __future__ import annotations

import argparse
import sys
import os

from handwriter.analyzer import analyze, AnalysisResult, CombinationScores
from handwriter.classifier import STYLE_NAMES


REFERENCE_SENTENCE = "what where who whom why how ? ! . ,"

BANNER = r"""
 _   _                 ___        __    _ _
| | | | __ _ _ __   __| \ \      / / __(_) |_ ___ _ __
| |_| |/ _` | '_ \ / _` |\ \ /\ / / '__| | __/ _ \ '__|
|  _  | (_| | | | | (_| | \ V  V /| |  | | ||  __/ |
|_| |_|\__,_|_| |_|\__,_|  \_/\_/ |_|  |_|\__\___|_|
"""


def print_scores(result: AnalysisResult) -> None:
    """Pretty-print the analysis result to stdout."""
    print("\n" + "=" * 62)
    print("  HANDWRITING STYLE ANALYSIS REPORT")
    print("=" * 62)
    print(f"  Image       : {result.image_path}")
    print(f"  Lines found : {result.num_lines}")
    print(f"  Letters segmented : {result.num_letters}")

    # Aggregate
    print("\n  --- Aggregate Style Scores (mean across all letters) ---")
    for name in STYLE_NAMES:
        bar_len = int(result.aggregate_scores.get(name, 0) * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"  {name:<14s} [{bar}] {result.aggregate_scores.get(name, 0):.4f}")

    # Per-letter breakdown
    print(f"\n  --- Per-Letter Breakdown ({result.num_letters} letters) ---")
    header = f"  {'Line':>4s} {'Idx':>4s} | " + " | ".join(f"{n[:6]:>6s}" for n in STYLE_NAMES) + " | Best"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ls in result.letter_scores:
        row = f"  {ls.line_index:4d} {ls.letter_index:4d} | "
        row += " | ".join(f"{ls.scores[n]:6.3f}" for n in STYLE_NAMES)
        tag = f" | {ls.best_style}"
        if ls.is_punctuation:
            tag += "  (punct)"
        row += tag
        print(row)

    # Letter Combination (bigram) breakdown
    if result.combination_scores:
        print(f"\n  --- Letter Combination Scores ({len(result.combination_scores)} pairs) ---")
        combo_header = (
            f"  {'Line':>4s} {'Pair':<8s} | "
            + " | ".join(f"{n[:6]:>6s}" for n in STYLE_NAMES)
            + " | Best"
        )
        print(combo_header)
        print("  " + "-" * (len(combo_header) - 2))

        for cs in result.combination_scores:
            row = f"  {cs.line_index:4d} {cs.label:<8s} | "
            row += " | ".join(f"{cs.scores[n]:6.3f}" for n in STYLE_NAMES)
            row += f" | {cs.best_style}"
            print(row)

    print("=" * 62 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HandWriter — Analyse handwriting style from an image.",
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to the handwriting image.",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        default=False,
        help="Output results as JSON instead of a formatted table.",
    )
    parser.add_argument(
        "--ink-color", "-c",
        type=str,
        choices=["dark", "blue"],
        default="blue",
        help="Ink colour: 'blue' for deep-blue ink on white paper (default), "
             "'dark' for standard dark ink.",
    )

    args = parser.parse_args()

    print(BANNER)

    if args.image is None:
        # Interactive mode
        print("  Welcome to HandWriter!")
        print("  ----------------------")
        print(f"\n  Step 1: Please write the following sentence on paper:\n")
        print(f'           "{REFERENCE_SENTENCE}"\n')
        print("  Step 2: Take a clear photo of your handwriting.")
        print("  Step 3: Enter the path to the photo below.\n")

        image_path = input("  Image path: ").strip().strip("'\"")
    else:
        image_path = args.image

    if not os.path.isfile(image_path):
        print(f"\n  Error: File not found -> {image_path}", file=sys.stderr)
        sys.exit(1)

    result = analyze(image_path, verbose=True, ink_color=args.ink_color)

    if args.json:
        print(result.to_json())
    else:
        print_scores(result)


if __name__ == "__main__":
    main()

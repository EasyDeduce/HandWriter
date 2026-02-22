# HandWriter

**HandWriter** is a handwriting analysis toolkit that segments handwritten text from an image, extracts stroke-level features from each letter, and classifies the writing style against six canonical handwriting types. The long-term goal is to learn a user's handwriting patterns and generate new text images in their style, packaged as a reusable Python deep learning library.

---

## Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Reference sentence prompt & image intake | Done |
| 2 | Letter segmentation + 6-style classification | Done |
| 3 | Bigram / transition pattern learning | Planned |
| 4 | Text-to-handwriting image generation | Planned |
| 5 | Packaged as a pip-installable DL library | Planned |

This repository currently implements **Phases 1 and 2**.

---

## Supported Handwriting Styles

Each segmented letter is scored on a **0-to-1 scale** against the following styles:

| Style | Key Characteristics |
|-------|---------------------|
| **Print** | Disconnected strokes, uniform width, upright posture, simple geometric forms. |
| **D'Nealian** | Slightly slanted, rounded letterforms, moderate connections, tail hooks on stems. |
| **Cursive** | Fully connected, flowing strokes, prominent loops, moderate-to-heavy slant. |
| **Italic Neat** | Consistent rightward slant, angular construction, clean strokes, slight connections. |
| **Spencerian** | Ornamental, heavy pressure variation (thick downstrokes / thin upstrokes), extreme slant. |
| **Calligraphy** | Dramatic thick-thin stroke contrast, decorative flourishes, artistic proportions. |

---

## Technical Approach

### 1. Image Preprocessing (`handwriter/preprocessor.py`)

- **Grayscale conversion** and **adaptive Gaussian thresholding** (Sauvola-style) to produce a clean binary image with ink pixels as foreground.
- **Morphological opening** for salt-and-pepper noise removal.
- **Line segmentation** via horizontal projection profile: rows are summed and contiguous bands of ink density are identified as text lines.
- **Letter segmentation** within each line via external contour detection (`cv2.findContours`), followed by left-to-right sorting and bounding-box merging for broken strokes that belong to the same character.

### 2. Feature Extraction (`handwriter/feature_extractor.py`)

Each segmented letter produces a **12-dimensional normalised feature vector**:

| # | Feature | How It Is Computed |
|---|---------|-------------------|
| 1 | **Slant angle** | Median angle of near-vertical Hough line segments. |
| 2 | **Stroke width mean** | Mean distance-transform value sampled along the morphological skeleton. |
| 3 | **Stroke width std** | Standard deviation of the above (captures pressure variation). |
| 4 | **Curvature mean** | Mean local cross-product curvature along the enclosing contour. |
| 5 | **Curvature std** | Spread of curvature values (angular vs. rounded forms). |
| 6 | **Aspect ratio** | Bounding-box width / height. |
| 7 | **Density** | Ink pixel count / bounding-box area. |
| 8 | **Loop count** | Number of enclosed holes detected via inverted-contour hierarchy. |
| 9 | **Connectivity** | Ink density in the leftmost and rightmost 12.5% of columns (linking strokes). |
| 10 | **Flourish score** | Fraction of ink that falls outside the convex hull of the main contour. |
| 11 | **Stroke length ratio** | Skeleton pixel count / bounding-box diagonal. |
| 12 | **Horizontal extent** | Fraction of columns containing ink. |

All values are clamped and linearly normalised to **[0, 1]**.

### 3. Style Classification (`handwriter/classifier.py`)

Each of the six styles is represented by:

- An **ideal feature profile** -- a 12-D vector capturing the expected feature values for that style.
- A **weight vector** -- controlling per-feature importance (e.g., *connectivity* matters heavily for Cursive, *stroke width std* matters heavily for Calligraphy).

The per-letter score is the **weighted cosine similarity** between the letter's feature vector and the style's ideal profile:

```
score(letter, style) = cos_sim(letter_vec * W_style,  ideal_style * W_style)
```

The result is clamped to [0, 1]. Aggregate scores across the full image are the arithmetic mean of per-letter scores.

### 4. Analysis Pipeline (`handwriter/analyzer.py`)

Orchestrates the full flow: `preprocess -> extract_features -> classify`. Returns an `AnalysisResult` object with per-letter and aggregate scores, serialisable to JSON.

---

## Project Structure

```
HandWriter/
├── handwriter/
│   ├── __init__.py            # Package marker
│   ├── preprocessor.py        # Image loading, binarisation, line & letter segmentation
│   ├── feature_extractor.py   # 12-D stroke feature computation
│   ├── classifier.py          # 6-style scoring via weighted cosine similarity
│   └── analyzer.py            # High-level orchestration pipeline
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Prerequisites

- **Python 3.12**
- A handwriting sample image (JPEG or PNG). Write the reference sentence on white paper with a dark pen and photograph it with reasonable lighting.

---

## Installation & Running

```bash
# 1. Clone or navigate to the project directory
cd HandWriter

# 2. Create a virtual environment (recommended)
python3.12 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Run in interactive mode (will prompt for the reference sentence and image path)
python main.py

# 4b. Run with a direct image path
python main.py --image /path/to/handwriting.jpg

# 4c. Get JSON output
python main.py --image /path/to/handwriting.jpg --json
```

---

## Example Output

```
  --- Aggregate Style Scores (mean across all letters) ---
  Print          [########......................] 0.2711
  D'Nealian      [###########...................] 0.3803
  Cursive        [################..............] 0.5417
  Italic Neat    [#########.....................] 0.3122
  Spencerian     [#######.......................] 0.2350
  Calligraphy    [######........................] 0.2104
```

Each letter also receives an individual breakdown, so you can see exactly which characters lean towards which style.

---

## Future Work (Phases 3-5)

- **Bigram pattern learning**: analyse letter transitions (e.g., *wh*, *at*) to capture how stroke connections change depending on neighbouring characters.
- **Handwriting generation**: given learned patterns, synthesise new text images that mimic the user's handwriting.
- **Library packaging**: publish as a pip-installable deep learning library with a clean Python API and pre-trained models.

---

## License

This project is currently unlicensed (private/research use).

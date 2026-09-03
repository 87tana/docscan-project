
"""
Generates eda_curated_notebook.ipynb — builds a 5-class curated subset
(form, invoice, handwritten, questionnaire, fallback) and runs EDA on it.
Stdlib only (no extra pip install). Run from your project root:
    python build_curated_notebook.py
"""
import json

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "id": f"cell-{len(cells)}", "metadata": {}, "source": src})


def code(src):
    cells.append({
        "cell_type": "code",
        "id": f"cell-{len(cells)}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    })


md("""\
# Docscan — Curated subset for healthcare digitization

**Target classes:** `form`, `invoice`, `handwritten`, `questionnaire` — plausible
document types in a healthcare intake/billing workflow (intake forms, medical
invoices, handwritten clinical notes, patient questionnaires). These use the
**full** original split counts — no subsampling: 1000 train / 200 val / 200
test per class.

**`fallback`:** everything else (the other 12 original classes), treated as a
normal 5th class sized to exactly match the target classes — **1000 train /
200 val / 200 test** — just pooled at random across those 12 classes combined
rather than belonging to one class. Result: a fully balanced 5-way set (5000
train, 1000 val, 1000 test — 7000 images total).

**Requirements:** `pillow`, `pandas`, `matplotlib`, `datasets==5.0.1`, numpy
(pulled in by pandas/matplotlib).
""")

code("""\
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_from_disk
from PIL import Image

%matplotlib inline

# Same validated palette as the main EDA notebook — categorical (train/val/test)
# uses fixed hue slots, plain magnitude uses one hue.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL = "#d03b3b"
INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SPLIT_COLORS = {"train": BLUE, "val": ORANGE, "test": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "grid.color": GRID, "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "figure.dpi": 100,
})

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
""")

md("## Load the full dataset")
code("""\
DATA_DIR = Path("data")
SPLITS = ["train", "val", "test"]

print("Loading dataset splits from disk...")
datasets = {split: load_from_disk(str(DATA_DIR / split)) for split in SPLITS}
class_names = datasets["train"].features["label"].names
print(f"{len(class_names)} classes available: {class_names}")
""")

md("""\
## Build the curated selection

Target classes take every row available in each split (1000/200/200,
matching the dataset's own per-class split sizes). `fallback` is a flat pool
sampled at random across all 12 non-target classes combined, sized to match
the target classes exactly per split (1000/200/200) — so it behaves like a
normal 5th class, not a 12x-oversized bucket. `row_index` points back into
the original `datasets[split]` object — no images are copied yet, just
indices.
""")
code("""\
TARGET_CLASSES = ["form", "invoice", "handwritten", "questionnaire"]
FALLBACK_LABEL = "fallback"
FALLBACK_SOURCE_CLASSES = [c for c in class_names if c not in TARGET_CLASSES]

# fallback is sized to match the target classes' own split sizes exactly
FALLBACK_N = {"train": 1000, "val": 200, "test": 200}


def build_curated_split(ds, split, seed):
    labels = ds["label"]
    indices_by_class = defaultdict(list)
    for i, lbl in enumerate(labels):
        indices_by_class[lbl].append(i)

    rng = random.Random(seed)
    rows = []

    # Target classes: take everything available in this split, no subsampling.
    for cls in TARGET_CLASSES:
        pool = indices_by_class[class_names.index(cls)]
        for i in pool:
            rows.append({"row_index": i, "label": cls, "source_class": cls})

    # Fallback: flat random pool across all 12 non-target classes combined,
    # sized to match the target-class split size (not per-class-stratified).
    fallback_pool = [
        (i, cls)
        for cls in FALLBACK_SOURCE_CLASSES
        for i in indices_by_class[class_names.index(cls)]
    ]
    n = FALLBACK_N[split]
    for i, cls in rng.sample(fallback_pool, min(n, len(fallback_pool))):
        rows.append({"row_index": i, "label": FALLBACK_LABEL, "source_class": cls})

    return rows


curated_rows = []
for split in SPLITS:
    for row in build_curated_split(datasets[split], split, RNG_SEED):
        row["split"] = split
        curated_rows.append(row)

curated_df = pd.DataFrame(curated_rows)
LABELS = TARGET_CLASSES + [FALLBACK_LABEL]


def fetch_curated_image(row):
    return datasets[row.split][int(row.row_index)]["image"]


print(f"Curated subset: {len(curated_df)} images across {len(SPLITS)} splits")
curated_df.head()
""")

md("""\
## 1. Overview & class balance

All 5 classes are sized identically per split (1000 train / 200 val / 200
test each) by construction — confirm that below rather than assuming it.
""")
code("""\
overview = curated_df.groupby(["split", "label"]).size().unstack(fill_value=0)[LABELS]
overview
""")
code("""\
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(LABELS))
width = 0.26
for i, split in enumerate(SPLITS):
    ax.bar(x + (i - 1) * width, overview.loc[split, LABELS], width, label=split, color=SPLIT_COLORS[split])
ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=20, ha="right")
ax.set_ylabel("Number of images")
ax.set_title("Curated subset: images per class, by split")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

for split in SPLITS:
    counts = overview.loc[split, LABELS]
    print(f"{split}: " + ", ".join(f"{lbl}={n}" for lbl, n in counts.items()) +
          (" -> balanced" if counts.nunique() == 1 else " -> NOT balanced"))
""")
code("""\
# fallback's 12-class composition — not stratified, so counts vary by chance
# (that's expected; it's a random pool, not a per-class quota).
fallback_breakdown = (
    curated_df[curated_df["label"] == FALLBACK_LABEL]
    .groupby(["split", "source_class"]).size().unstack(fill_value=0)
)
fallback_breakdown
""")

md("""\
## 2. Decode the curated images

7,000 images total (5000/1000/1000) — decode everything and compute
per-image stats in one pass (dimensions, grayscale brightness for the
blank-scan check, and a perceptual hash for duplicates).
""")
code("""\
def average_hash(img, hash_size=8):
    small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float64)
    bits = (arr > arr.mean()).flatten()
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


widths, heights, aspects = [], [], []
modes, formats = [], []
gray_means, gray_stds, ahashes = [], [], []
bad_images = []

for row in curated_df.itertuples():
    try:
        img = fetch_curated_image(row)
        w, h = img.size
        gray = np.asarray(img.convert("L"), dtype=np.float64)
        widths.append(w); heights.append(h); aspects.append(round(w / h, 3))
        modes.append(img.mode); formats.append(img.format)
        gray_means.append(gray.mean()); gray_stds.append(gray.std())
        ahashes.append(average_hash(img))
    except Exception as e:
        bad_images.append((row.split, row.row_index, str(e)))
        widths.append(None); heights.append(None); aspects.append(None)
        modes.append(None); formats.append(None)
        gray_means.append(None); gray_stds.append(None); ahashes.append(None)

curated_df["width"] = widths
curated_df["height"] = heights
curated_df["aspect_ratio"] = aspects
curated_df["mode"] = modes
curated_df["format"] = formats
curated_df["gray_mean"] = gray_means
curated_df["gray_std"] = gray_stds
curated_df["ahash"] = ahashes

print(f"Decoded {len(curated_df) - len(bad_images)} images ({len(bad_images)} failed)")
curated_df[["width", "height", "aspect_ratio", "gray_mean", "gray_std"]].describe()
""")
code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, title in zip(axes, ["width", "height", "aspect_ratio"],
                           ["Width (px)", "Height (px)", "Aspect ratio (w / h)"]):
    ax.hist(curated_df[col].dropna(), bins=30, color=BLUE, edgecolor=SURFACE)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# Printed, not plotted — a single-value bar chart says nothing a print doesn't.
print("Image mode counts:")
print(curated_df["mode"].value_counts())
print("\\nImage format counts:")
print(curated_df["format"].astype(str).value_counts())
""")

md("## 3. Data quality checks")
code("""\
if bad_images:
    print(f"{len(bad_images)} images failed to decode:")
    for split, idx, err in bad_images[:20]:
        print(f"  {split}[{idx}]: {err}")
else:
    print("No corrupted images in the curated subset.")
""")
code("""\
BLANK_MEAN_THRESHOLD = 250
BLANK_STD_THRESHOLD = 8
curated_df["possibly_blank"] = (
    (curated_df["gray_mean"] > BLANK_MEAN_THRESHOLD) | (curated_df["gray_std"] < BLANK_STD_THRESHOLD)
)
n_blank = int(curated_df["possibly_blank"].sum())
print(f"{n_blank} of {len(curated_df)} images ({n_blank / len(curated_df) * 100:.1f}%) look possibly blank")
curated_df.groupby("label")["possibly_blank"].sum()
""")
code("""\
hash_to_rows = defaultdict(list)
for row in curated_df.itertuples():
    if row.ahash is not None:
        hash_to_rows[row.ahash].append(row.Index)
dup_groups = {h: idxs for h, idxs in hash_to_rows.items() if len(idxs) > 1}
n_dup_images = sum(len(v) for v in dup_groups.values())
print(f"{len(dup_groups)} near-duplicate groups ({n_dup_images} images total)")
""")

md("## 4. Visual exploration by class")
code("""\
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, lbl in zip(axes.flat, LABELS):
    row = curated_df[curated_df["label"] == lbl].sample(1, random_state=RNG_SEED).iloc[0]
    img = fetch_curated_image(row)
    ax.imshow(img, cmap="gray" if img.mode == "L" else None)
    ax.set_title(lbl, fontsize=11)
    ax.axis("off")
axes.flat[-1].axis("off")
plt.suptitle("One example per class")
plt.tight_layout()
plt.show()
""")
code("""\
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
xmax, ymax = curated_df["width"].max() * 1.05, curated_df["height"].max() * 1.05
for ax, lbl in zip(axes.flat, LABELS):
    sub = curated_df[curated_df["label"] == lbl]
    ax.scatter(sub["width"], sub["height"], s=10, alpha=0.4, color=BLUE, linewidths=0)
    ax.set_title(lbl, fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
axes.flat[-1].axis("off")
fig.supxlabel("Width (px)")
fig.supylabel("Height (px)")
plt.tight_layout()
plt.show()
""")
code("""\
def show_class_examples(label, n=6):
    sub = curated_df[curated_df["label"] == label]
    sample = sub.sample(min(n, len(sub)), random_state=RNG_SEED)
    fig, axes = plt.subplots(1, len(sample), figsize=(15, 4))
    if len(sample) == 1:
        axes = [axes]
    for ax, row in zip(axes, sample.itertuples()):
        img = fetch_curated_image(row)
        ax.imshow(img, cmap="gray" if img.mode == "L" else None)
        ax.set_title(f"{row.width}x{row.height}" if row.width else "?", fontsize=9)
        ax.axis("off")
    plt.suptitle(f"Examples: {label}")
    plt.tight_layout()
    plt.show()


show_class_examples("handwritten")
show_class_examples("fallback")
""")

md("## 5. Summary")
code("""\
print("=" * 60)
print("CURATED SUBSET SUMMARY")
print("=" * 60)
print(f"Target classes: {TARGET_CLASSES}")
print(f"Fallback pools from: {FALLBACK_SOURCE_CLASSES}")
print("Per split: all 5 classes balanced at 1000 train / 200 val / 200 test")
print(f"Total images: {len(curated_df)}")
print(f"Decode failures: {len(bad_images)}")
print(f"Possibly-blank: {n_blank} ({n_blank / len(curated_df) * 100:.1f}%)")
print(f"Near-duplicate groups: {len(dup_groups)} ({n_dup_images} images)")
""")

md("""\
## Optional: save the selection

Only indices + labels — the actual pixels stay in `data/{split}`. This CSV
lets you reload the exact same 7,000 images later without re-sampling.
""")
code("""\
curated_df.drop(columns=["ahash"]).to_csv("curated_subset.csv", index=False)
print(f"Saved curated_subset.csv ({len(curated_df)} rows)")
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("eda_curated_notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Wrote eda_curated_notebook.ipynb with {len(cells)} cells")
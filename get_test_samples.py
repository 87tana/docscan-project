import pandas as pd
from datasets import load_from_disk
import os

OUTPUT_DIR = "test_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("data/curated_subset.csv")
test_rows = df[df["split"] == "test"]

ds = load_from_disk("data/test")

# Grab 3 fresh samples per class
for cls in ["form", "invoice", "handwritten", "questionnaire", "fallback"]:
    cls_rows = test_rows[test_rows["label"] == cls].sample(n=3, random_state=42)
    for i, row in enumerate(cls_rows.itertuples()):
        sample = ds[row.row_index]
        filename = f"{OUTPUT_DIR}/{cls}_{i+1}.png"
        sample["image"].save(filename)
        print(f"Saved: {filename}")

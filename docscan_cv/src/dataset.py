import pandas as pd
from datasets import load_from_disk
from torch.utils.data import Dataset


LABELS = ["form", "invoice", "handwritten", "questionnaire", "fallback"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}


class CuratedDocDataset(Dataset):
    """Reads curated_subset.csv and pulls the matching image from the
    original arrow dataset for one split (train/val/test)."""

    def __init__(self, csv_path, data_dir, split, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.ds = load_from_disk(f"{data_dir}/{split}")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.ds[int(row["row_index"])]["image"].convert("RGB")
        label = LABEL_TO_IDX[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label

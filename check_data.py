from datasets import load_from_disk

for split in ["train", "val", "test"]:
    ds = load_from_disk(f"data/{split}")
    print(f"\n=== {split} ===")
    print("num rows:", len(ds))
    print("features:", ds.features)
    print("first example keys:", ds[0].keys())

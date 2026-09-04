import argparse
import random

import numpy as np
import torch
import yaml
import mlflow
from torch.utils.data import DataLoader

from src.dataset import CuratedDocDataset, LABELS
from src.transforms import build_transform
from src.models import build_model
from src.engine import train_one_epoch, evaluate
from src import mlflow_utils


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, image_size, mean, std = build_model(config)
    model = model.to(device)

    train_transform = build_transform(image_size, mean, std, train=True)
    eval_transform = build_transform(image_size, mean, std, train=False)

    train_ds = CuratedDocDataset(config["curated_csv"], config["data_dir"], "train", train_transform)
    val_ds = CuratedDocDataset(config["curated_csv"], config["data_dir"], "val", eval_transform)
    test_ds = CuratedDocDataset(config["curated_csv"], config["data_dir"], "test", eval_transform)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])



    criterion = torch.nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"])

    best_val_accuracy = 0.0
    checkpoint_path = f"{config['output_dir']}_best.pt"

    with mlflow_utils.start_run(config):
        mlflow_utils.log_params(config)

        for epoch in range(config["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_accuracy, _, _, _ = evaluate(model, val_loader, criterion, device)

            mlflow_utils.log_epoch_metrics(epoch, train_loss, val_loss, val_accuracy)
            print(f"epoch {epoch+1}/{config['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), checkpoint_path)

        model.load_state_dict(torch.load(checkpoint_path))
        test_loss, test_accuracy, test_cm, _, _ = evaluate(model, test_loader, criterion, device)
        print(f"test_loss={test_loss:.4f}  test_acc={test_accuracy:.4f}")

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow_utils.log_confusion_matrix(test_cm, LABELS)
        mlflow_utils.log_checkpoint(checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)

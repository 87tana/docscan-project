import matplotlib.pyplot as plt
import mlflow


def start_run(config):
    mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", "sqlite:///mlruns.db"))
    mlflow.set_experiment(config["mlflow_experiment_name"])
    return mlflow.start_run(run_name=config.get("run_name"))


def log_params(config):
    mlflow.log_params({
        "model_name": config["model_name"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "lr": config["lr"],
        "freeze_backbone": config.get("freeze_backbone", True),
        "seed": config["seed"],
    })


def log_epoch_metrics(epoch, train_loss, val_loss, val_accuracy):
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }, step=epoch)


def log_confusion_matrix(cm, class_names, filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    mlflow.log_artifact(filename)


def log_checkpoint(path):
    mlflow.log_artifact(path)

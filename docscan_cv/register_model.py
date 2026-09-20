import sys
import os
import yaml
import torch
import mlflow

# Let this script import from src/ (models.py etc.)
sys.path.append(os.path.dirname(__file__))

from src.models import build_model

# --- Config ---
PROJECT_ROOT = "/home/tannaz/Documents/Projects/Docscan_project"
MLFLOW_LOCAL_DIR = os.path.join(PROJECT_ROOT, "mlflow_local")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "dit_base.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

print("Loaded config for model:", config["model_name"])

# --- MLflow tracking setup ---
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(MLFLOW_LOCAL_DIR, 'mlruns.db')}")

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("docscan-cv-v2")

print("Tracking URI:", mlflow.get_tracking_uri())
print("Experiment ID:", experiment.experiment_id)

# --- Find the best run ---
best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_accuracy DESC"],
    max_results=1
)[0]

run_id = best_run.info.run_id
accuracy = best_run.data.metrics["test_accuracy"]

print("Best run_id:", run_id)
print("Best accuracy:", accuracy)


# --- Create (or reuse) a LOCAL experiment for registration, with a real local artifact path ---
LOCAL_ARTIFACT_ROOT = os.path.join(MLFLOW_LOCAL_DIR, "mlartifacts_registration")

local_exp_name = "docscan-cv-v2-local"
local_exp = client.get_experiment_by_name(local_exp_name)

if local_exp is None:
    local_exp_id = client.create_experiment(
        name=local_exp_name,
        artifact_location=f"file://{LOCAL_ARTIFACT_ROOT}"
    )
else:
    local_exp_id = local_exp.experiment_id

mlflow.set_experiment(experiment_id=local_exp_id)

print("Using local experiment:", local_exp_name, "| id:", local_exp_id)

# --- Load the checkpoint into the model architecture ---
model, image_size, mean, std = build_model(config)

checkpoint_path = os.path.join(
    MLFLOW_LOCAL_DIR, "mlartifacts", run_id, "artifacts", "dit_base_best.pt"
)

state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

print("Checkpoint loaded from:", checkpoint_path)
print("Model type:", type(model))


# --- Log model properly to MLflow, then register it ---
example_input = torch.randn(1, 3, image_size, image_size)

with mlflow.start_run(run_name="dit_base_registration") as run:
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        input_example=example_input.numpy(),
    )
    logged_model_uri = f"runs:/{run.info.run_id}/model"

print("Logged model URI:", logged_model_uri)

registered_model = mlflow.register_model(
    model_uri=logged_model_uri,
    name="docscan-classifier"
)

print("Registered as:", registered_model.name)
print("Version:", registered_model.version)

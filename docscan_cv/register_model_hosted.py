from dotenv import load_dotenv
load_dotenv()

import os
import sys
import yaml
import torch
import mlflow

sys.path.append(os.path.dirname(__file__))
from src.models import build_model

# --- Point MLflow at the HOSTED registry (RDS + S3) ---
RDS_HOST = os.environ["RDS_HOST"]
RDS_PORT = os.environ["RDS_PORT"]
RDS_USER = os.environ["RDS_USER"]
RDS_PASSWORD = os.environ["RDS_PASSWORD"]
RDS_DB_NAME = os.environ["RDS_DB_NAME"]

tracking_uri = f"postgresql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("docscan-cv-hosted")

# --- Load the same trained checkpoint we used before (still local, that's fine) ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "dit_base.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = "/home/tannaz/Documents/Projects/Docscan_project"
checkpoint_path = os.path.join(
    PROJECT_ROOT, "mlflow_local", "mlartifacts",
    "30cf5aab14dd40b7b02d9a74b7dd00a7", "artifacts", "dit_base_best.pt"
)

model, image_size, mean, std = build_model(config)
state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

print("Checkpoint loaded.")

# --- Log to the hosted registry (this uploads to S3!) ---
example_input = torch.randn(1, 3, image_size, image_size)

with mlflow.start_run(run_name="dit_base_hosted_registration") as run:
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

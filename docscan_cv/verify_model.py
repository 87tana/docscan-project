import mlflow
import os

PROJECT_ROOT = "/home/tannaz/Documents/Projects/Docscan_project"
MLFLOW_LOCAL_DIR = os.path.join(PROJECT_ROOT, "mlflow_local")

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(MLFLOW_LOCAL_DIR, 'mlruns.db')}")

client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='docscan-classifier'")

if versions:
    for v in versions:
        print(f"{v.name} | version: {v.version} | status: {v.status}")
else:
    print("Model not found in registry!")

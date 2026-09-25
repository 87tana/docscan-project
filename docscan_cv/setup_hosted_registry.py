from dotenv import load_dotenv
load_dotenv()

import os
import mlflow

RDS_HOST = os.environ["RDS_HOST"]
RDS_PORT = os.environ["RDS_PORT"]
RDS_USER = os.environ["RDS_USER"]
RDS_PASSWORD = os.environ["RDS_PASSWORD"]
RDS_DB_NAME = os.environ["RDS_DB_NAME"]

tracking_uri = f"postgresql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}"
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "docscan-cv-hosted"
artifact_location = "s3://docscan-mlflow-artifacts/"

client = mlflow.tracking.MlflowClient()
existing = client.get_experiment_by_name(experiment_name)

if existing is None:
    exp_id = client.create_experiment(name=experiment_name, artifact_location=artifact_location)
    print(f"Created experiment: {experiment_name} (id={exp_id})")
else:
    print(f"Experiment already exists: {experiment_name} (id={existing.experiment_id})")

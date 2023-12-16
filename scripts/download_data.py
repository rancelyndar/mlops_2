import os
import gdown
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/home/sergey/flow/project/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("data_download")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def download_data(id):
    gdown.download(id=id, output=os.path.join(BASE_DIR, "datasets/churn.csv"))

if __name__ == "__main__":
    with mlflow.start_run():
        download_data('1trL3K4xLpCKyp5arYrVaWDMn3jjt1jVL')
    mlflow.log_artifact(local_path="/home/sergey/flow/project/scripts/download_data.py",
                        artifact_path="download_data artifact")
    mlflow.end_run()

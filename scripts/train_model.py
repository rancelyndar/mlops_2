import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_model(df: pd.DataFrame):
    over = SMOTE(sampling_strategy=1)

    f1 = df.drop(columns=['Exited']).iloc[:, :].values
    t1 = df.iloc[:, -1].values

    f1, t1 = over.fit_resample(f1, t1)
    x_train, x_test, y_train, y_test = train_test_split(f1, t1, test_size=0.22, random_state=100)
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(x_train,y_train)
        mlflow.log_artifact(local_path="/home/sergey/flow/project/scripts/train_model.py",
                            artifact_path="train_model artifact")
        mlflow.end_run()
    print('Test accuracy is :{:.6f}'.format(accuracy_score(y_test,model.predict(x_test))))

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(BASE_DIR, "datasets/churn_prepared.csv"))
    train_model(df)
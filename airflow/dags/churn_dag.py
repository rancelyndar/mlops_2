from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    "owner": "admin",
    "start_date": dt.datetime(2023, 12, 16, 15, 15),
    "retries": 5,
    "retry_delays": dt.timedelta(seconds=30),
    "depends_on_past": False,
}

with DAG(
    "Churn prediction training",
    description="Churn prediction",
    schedule_interval="*/5 * * * *",
    default_args=args,
    tags=["churn", "classification"],
) as dag:
    download_data = BashOperator(
        task_id="download_data",
        bash_command="python3 /home/sergey/flow/project/scripts/download_data.py",
        dag=dag,
    )
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="python3 /home/sergey/flow/project/scripts/prepare_data.py",
        dag=dag,
    )
    train_model = BashOperator(
        task_id="train_model",
        bash_command="python3 /home/sergey/flow/project/scripts/train_model.py",
        dag=dag,
    )
    download_data >> prepare_data >> train_model

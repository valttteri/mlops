import minio
import time
import subprocess
import mlflow
from pathlib import Path
import kfp


def clean_mlflow(mlflow_client: mlflow.MlflowClient, experiment_name: str):
    """
    Clean an MLflow experiment by deleting all the runs and the registered model associated with the runs
    """
    mlflow_exp = mlflow_client.get_experiment_by_name(experiment_name)
    mlflow_runs = mlflow_client.search_runs(experiment_ids=[mlflow_exp.experiment_id])
    for mlflow_run in mlflow_runs:
        model_versions = mlflow_client.search_model_versions(
            f"run_id='{mlflow_run.info.run_id}'"
        )
        # Delete the registered model corresponding to the run if any
        if len(model_versions) > 0:
            mv = model_versions[0]
            mlflow_client.delete_model_version(mv.name, mv.version)
        # Permanently delete the run
        mlflow_client.delete_run(mlflow_run.info.run_id)
        subprocess.run(
            [
                str(Path(__file__).parent / "delete_mlflow_run.sh"),
                mlflow_run.info.run_id,
            ]
        )


def clean_kfp_experiment(kfp_client: kfp.Client, kfp_experiment_name: str):
    """
    Delete a KFP experiment and all its KFP runs
    """
    experiment_id = kfp_client.get_experiment(
        experiment_name=kfp_experiment_name
    ).experiment_id

    # Get all runs under the experiment
    response = kfp_client.list_runs(experiment_id=experiment_id)

    # Delete all the runs
    for kfp_run in response.runs:
        kfp_client.runs.delete_run(run_id=kfp_run.run_id)

    # Delete the experiment
    kfp_client.delete_experiment(experiment_id=experiment_id)


def _clean_bucket_helper(
    minio_client: minio.Minio, bucket_name: str, prefix: str = None
):
    # Clean bucket from minio
    objects_to_delete = minio_client.list_objects(
        bucket_name=bucket_name, prefix=prefix, recursive=True
    )
    for obj in objects_to_delete:
        minio_client.remove_object(bucket_name=bucket_name, object_name=obj.object_name)


def clean_kfp_bucket(minio_client: minio.Minio, kfp_run_id: str):
    """
    Clean the KFP bucket from Minio by deleting all the objects associated with a KFP run
    """
    with subprocess.Popen(
        ["kubectl", "-n", "kubeflow", "port-forward", "svc/minio-service", "5005:9000"],
        stdout=True,
    ) as proc:
        try:
            time.sleep(10)
            _clean_bucket_helper(
                minio_client=minio_client,
                bucket_name="mlpipeline",
                prefix=f"v2/artifacts/bike-demand-pipeline/{kfp_run_id}",
            )
            _clean_bucket_helper(
                minio_client=minio_client, bucket_name="mlpipeline", prefix="artifacts"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            raise e
        finally:
            proc.terminate()

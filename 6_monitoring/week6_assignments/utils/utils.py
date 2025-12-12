# Helper functions

import requests
from typing import List
from time import sleep
import simplejson
import pandas as pd
from lightgbm import LGBMRegressor
import mlflow
from mlflow import MlflowClient
import os
from typing import Dict, Any, NamedTuple, Optional
from collections import namedtuple
import minio
import subprocess
import time

from evidently.ui.dashboards import (
    ReportFilter,
    DashboardPanelTestSuiteCounter,
    CounterAgg,
    DashboardPanelPlot,
    PanelValue,
    PlotType,
    DashboardConfig,
)
from evidently.renderers.html_widgets import WidgetSize
from evidently import metrics
from evidently.ui.remote import RemoteWorkspace
from evidently.ui.workspace import Workspace
from evidently.ui.base import Project

import kfp

from utils.config import (
    MLFLOW_S3_ENDPOINT_URL,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    MLFLOW_TRACKING_URI,
    STAGE_TAG_KEY,
    PROD_STAGE,
    REGISTERED_MODEL_NAME,
    MLFLOW_EXPERIMENT_NAME,
)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = MlflowClient()


def get_model_info() -> Optional[
    NamedTuple(
        "ModelInfo",
        [
            ("model_version", str),
            ("model_s3_uri", str),
        ],
    )
]:
    """
    Get the version and S3 URI of the model that is in the production stage
    """
    model_info = namedtuple("ModelInfo", ["model_version", "model_s3_uri"])

    mv_arr = mlflow_client.search_model_versions(
        filter_string=f"name='{REGISTERED_MODEL_NAME}' and tag.{STAGE_TAG_KEY}='{PROD_STAGE}'"
    )
    if len(mv_arr) > 0:
        mv = mv_arr[0]
        return model_info(mv.version, mv.source)

    return None


def train(
    train_x: Optional[pd.DataFrame] = None,
    train_y: Optional[pd.DataFrame] = None,
    params: Optional[Dict[str, Any]] = None,
) -> NamedTuple(
    "ModelInfo",
    [
        ("model_version", str),
        ("model_s3_uri", str),
    ],
):
    """
    Train a LightGBM regression model and register it to MLflow, then add a {"stage": "Production"} tag to the registered model version.
    Args:
        train_x: Features
        train_y: Target
        params: Hyperparameters for the model
    Returns:
        NamedTuple: Model version and its S3 URI
    """
    model_info = namedtuple("ModelInfo", ["model_version", "model_s3_uri"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    model = LGBMRegressor(**params, verbose=-1)

    with mlflow.start_run() as run:
        model.fit(train_x, train_y)

        for key, val in params.items():
            mlflow.log_param(key, val)

        model_artifact_path = "lgbm-house"
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path=model_artifact_path,
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        model_s3_uri = mlflow.get_artifact_uri(artifact_path=model_artifact_path)
        model_version = mlflow_client.get_latest_versions(
            REGISTERED_MODEL_NAME, stages=["None"]
        )[0].version
        mlflow_client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=model_version,
            key=STAGE_TAG_KEY,
            value=PROD_STAGE,
        )
        return model_info(model_version, model_s3_uri)


def send_requests(model_name: str, input: List, count: int):
    """
    Send some requests to an inference service

    Args:
        model_name (str): Name of the inference service
        input: Features based on which the inference service make predictions
        count: Number of requests
    """
    request_data = {"parameters": {"content_type": "pd"}, "inputs": input}
    headers = {}
    kserve_gateway_url = "http://kserve-gateway.local:30200"

    headers["Host"] = f"{model_name}.kserve-inference.example.com"
    headers["content-type"] = "application/json"

    url = f"{kserve_gateway_url}/v2/models/{model_name}/infer"
    if count == 1:
        res = requests.post(
            url, data=simplejson.dumps(request_data, ignore_nan=True), headers=headers
        )
        return res

    for i in range(count):
        requests.post(
            url, data=simplejson.dumps(request_data, ignore_nan=True), headers=headers
        )
        sleep(0.5)
        print(f"{i+1} requests have been sent", flush=True)


def init_evidently_project(
    workspace: Workspace | RemoteWorkspace, project_name: str
) -> Project:
    """
    Create a Project to a Workspace
    Args:
        workspace: An Evidently Workspace
        project_name: Name of the Project
    """
    # Delete any projects whose name is the given project_name to avoid duplicated projects
    for project in workspace.search_project(project_name=project_name):
        workspace.delete_project(project_id=project.id)

    # Create a project at Evidently
    project = workspace.create_project(name=project_name)

    # Create a dashboard
    project.dashboard = DashboardConfig(name=project_name, panels=[])

    project.dashboard.add_panel(
        DashboardPanelTestSuiteCounter(title="MAE", agg=CounterAgg.LAST),
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="MAE",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="RegressionQualityMetric",
                    field_path=metrics.RegressionQualityMetric.fields.current.mean_abs_error,
                    legend="MAE",
                ),
            ],
            plot_type=PlotType.LINE,
            size=WidgetSize.FULL,
        )
    )

    project.save()
    return project


def check_snapshots(
    year: int, model_version: str, report_snapshots: List, test_snapshots: List
):
    """
    Test if the local Evidently workspace contains the correct reports and test suites
    Args:
        year: The year of the monitoring results
        model_version: The version of the registered model that is being monitored
        report_snapshots: The snapshots of the reports
        test_snapshots: The snapshots of the test suites
    """
    # There should also be 4 Reports
    assert len(report_snapshots) == 4

    # Check if each Report uses the required Presets and has the required tags
    report_snapshots.sort(key=lambda s: sorted(s["tags"])[0])
    for i in range(4):
        snapshot = report_snapshots[i]
        assert set(snapshot["metadata"]["metric_presets"]) == set(
            ["RegressionPreset", "TargetDriftPreset", "DataDriftPreset"]
        ), f"Report {i+1} does not use the required presets"
        assert set(snapshot["tags"]) == set(
            [f"{year}-quarter{i+1}", f"{REGISTERED_MODEL_NAME}-{model_version}"]
        ), f"Report {i+1} does not have the required tags"

    # There should also be 4 Test Suites
    assert len(test_snapshots) == 4

    # Check if each Test Suite has the required tags
    test_snapshots.sort(key=lambda s: sorted(s["tags"])[0])
    for i in range(4):
        snapshot = test_snapshots[i]
        assert set(snapshot["tags"]) == set(
            [f"{year}-quarter{i+1}", f"{REGISTERED_MODEL_NAME}-{model_version}"]
        ), f"Test Suite {i+1} does not have the required tags"


def is_being_graded():
    """
    Returns True if the notebook is being executed by the auto-grading tool.
    """
    env = os.environ.get("NBGRADER_EXECUTION")
    return env == "autograde" or env == "validate"


def delete_kfp_exp(experiment_name: str):
    """
    Delete a KFP experiment and all its runs
    """
    kfp_client = kfp.Client()
    experiment_id = kfp_client.get_experiment(
        experiment_name=experiment_name
    ).experiment_id

    # Get all runs under the experiment
    response = kfp_client.list_runs(experiment_id=experiment_id)

    # Delete all the runs
    for run in response.runs:
        kfp_client.runs.delete_run(run_id=run.run_id)

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
                prefix=f"v2/artifacts/monitoring-pipeline/{kfp_run_id}",
            )
            _clean_bucket_helper(
                minio_client=minio_client, bucket_name="mlpipeline", prefix="artifacts"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            raise e
        finally:
            proc.terminate()

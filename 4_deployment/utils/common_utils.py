"""
Train a new LightGBM model for predicting bike sharing demand
This is the same as the assignment in Week1, we just change the hyperparameters of the model and skip the part that uses Deepchecks to perform offline model evaluation
"""

import pandas as pd
import os
from typing import Dict, Tuple, Any

from lightgbm import LGBMRegressor
import xgboost as xgb

import mlflow
from mlflow import MlflowClient

# mlflow configuration
MLFLOW_S3_ENDPOINT_URL = "http://mlflow-minio.local"
MLFLOW_TRACKING_URI = "http://mlflow-server.local"
AWS_ACCESS_KEY_ID = "minioadmin"
AWS_SECRET_ACCESS_KEY = "minioadmin"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_CLIENT = MlflowClient()

# dataset URL
DATASET_URL = "https://raw.githubusercontent.com/yumoL/mlops_eng_course_datasets/master/intro/bike-demanding/train_full.csv"

# name of the target column
TARGET = "count"

# registered model names for the LightGBM and XGBoost models
REGISTERED_LGBM_MODEL_NAME = "Week4LgbmBikeDemand"
REGISTERED_XGB_MODEL_NAME = "Week4XgbBikeDemand"


def pull_data() -> pd.DataFrame:
    """
    Download the data set from a given url
    Args:
        dataset_url: dataset url
    Returns:
        A Pandas DataFrame of the dataset
    """
    input_df = pd.read_csv(DATASET_URL)
    return input_df


def preprocess(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data and split them into a training and a test dataset
    Args:
        input_df: The DataFrame of the whole dataset
    Returns:
        a tuple of two DataFrames, one for training and another for testing dataset
    """
    input_df["datetime"] = pd.to_datetime(input_df["datetime"])

    # create hour, day and month variables from datetime column
    input_df["hour"] = input_df["datetime"].dt.hour
    input_df["day"] = input_df["datetime"].dt.day
    input_df["month"] = input_df["datetime"].dt.month

    # drop datetime, casual and registered columns
    input_df.drop(["datetime", "casual", "registered"], axis=1, inplace=True)

    # split the original dataset into a training and a test dataset
    horizon = 24 * 7
    train, test = input_df.iloc[:-horizon, :], input_df.iloc[-horizon:, :]
    return train, test


def train_model(
    model_type: str,
    model_params: Dict[str, Any],
    freshness_tag: str,
) -> str:
    """
    Train a new model and log it to MLflow
    Args:
        model_type: The type of the model, either "lgbm" or "xgb"
        model_params: The hyperparameters of the model
        freshness_tag: A tag of the model. Later it will be used to check whether a model is existing or not to avoid duplicated training
    Returns:
        The URI of the trained model
    """
    input_df = pull_data()
    train, _ = preprocess(input_df)

    train_x = train.drop([TARGET], axis=1)
    train_y = train[[TARGET]]

    model = None
    artifact_path = ""
    mlflow_exp_name = ""
    registered_model_name = ""
    if model_type == "lgbm":
        model = LGBMRegressor(**model_params)
        model.fit(train_x, train_y)
        registered_model_name = REGISTERED_LGBM_MODEL_NAME
        mlflow_exp_name = "week4-lgbm-bike-demand"
        artifact_path = "lgbm-bike"

    elif model_type == "xgb":
        dtrain = xgb.DMatrix(train_x, label=train_y)
        model = xgb.train(model_params, dtrain)
        registered_model_name = REGISTERED_XGB_MODEL_NAME
        mlflow_exp_name = "week4-xgb-bike-demand"
        artifact_path = "xgb-bike"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    mlflow.set_experiment(mlflow_exp_name)

    with mlflow.start_run() as run:
        for hyperparam_name, value in model_params.items():
            mlflow.log_param(hyperparam_name, value)

        if model_type == "lgbm":
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
        elif model_type == "xgb":
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                model_format="bst",
            )

        model_info = MLFLOW_CLIENT.get_latest_versions(registered_model_name)[0]
        MLFLOW_CLIENT.set_model_version_tag(
            name=registered_model_name,
            version=model_info.version,
            key="freshness",
            value=freshness_tag,
        )

        model_artifact_uri = f"{mlflow.get_artifact_uri(artifact_path=artifact_path)}"

        print(f"The trained model is located at {model_artifact_uri}")

        return model_artifact_uri


def train(model_type: str, model_params: Dict[str, Any], freshness_tag: str) -> str:
    """
    Train a new model or use the existing model if it exists
    Args:
        model_type: The type of the model, either "lgbm" or "xgb"
        model_params: The hyperparameters of the model
        freshness_tag: A tag of the model. A model with the same tag will be reused if it exists
    Returns:
        The S3 URI of the model
    """
    registered_model_name = (
        REGISTERED_LGBM_MODEL_NAME
        if model_type == "lgbm"
        else REGISTERED_XGB_MODEL_NAME
    )
    search_str = f"name='{registered_model_name}' AND tag.freshness='{freshness_tag}'"
    mv_arr = MLFLOW_CLIENT.search_model_versions(search_str)

    if len(mv_arr) == 0:
        print("No model found, start training...")
        return train_model(
            model_type=model_type,
            model_params=model_params,
            freshness_tag=freshness_tag,
        )

    print(f"Model found, skip training and use the existing model {mv_arr[0].source}")
    return mv_arr[0].source

# if __name__ == "__main__":
#     # LGBM hyperparameters
#     lgbm_params = {
#         "n_estimators": 100,
#         "learning_rate": 0.1,
#         "max_depth": 5,
#         "num_leaves": 32,
#         "objective": "regression",
#         "boosting_type": "gbdt",
#         "metric": "rmse",
#     }

#     # XGBoost hyperparameters
#     xgb_params = {
#         "max_depth": 6,
#         "eta": 0.3,
#         "objective": "reg:squarederror",
#         "eval_metric": "rmse",
#         "n_estimators": 100,
#     }

#     # train LGBM model
#     train("lgbm", lgbm_params, "new")

#     # train XGBoost model
#     train("xgb", xgb_params, "old")
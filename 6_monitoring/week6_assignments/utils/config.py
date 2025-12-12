# Fixed variables used in assignments

# mlflow
MLFLOW_S3_ENDPOINT_URL = "http://mlflow-minio.local"
MLFLOW_TRACKING_URI = "http://mlflow-server.local"
AWS_ACCESS_KEY_ID = "minioadmin"
AWS_SECRET_ACCESS_KEY = "minioadmin"
MLFLOW_EXPERIMENT_NAME = "week6-lgbm-house-price"

# The name used when registering the model to MLflow, will be
# passed to the "registered_model_name" parameter of mlflow.lightgbm.log_model()
REGISTERED_MODEL_NAME = "Week6LgbmHousePrice"

# The tag key to indicate which stage (e.g. production) the model is in
STAGE_TAG_KEY = "stage"

# The tag value to indicate the model is in the production stage
PROD_STAGE = "Production"

# monitoring
# URL of the remote Evidently monitor
EVIDENTLY_MONITOR_URL = "http://evidently-monitor-ui.local"

# Directory where model inputs(features received on production) and outputs(predictions) are saved
INPUTS_OUTPUTS_LOCAL_DIR_NAME = "inputs_outputs"

# Directory where ground truth is saved
GROUND_TRUTH_LOCAL_DIR_NAME = "ground_truth"

# Bucket names used to save the inputs-outputs and ground truth in MinIO
INPUTS_OUTPUTS_BUCKET_NAME = "inputs-outputs"
GROUND_TRUTH_BUCKET_NAME = "ground-truth"

# Directory where the feature-engineered data (data preprocessed by the etl function from the second week) is saved
FEATURE_STORE_DIR_NAME = "feature_store_quarterly"

# The column mapping used by Evidently
COLUMN_MAPPING_DICT = {
    "numerical_features": [
        "yr_built",
        "bedrooms",
        "postcode",
        "area",
        "bathrooms",
        "condition",
        "grade",
        "sqft_living",
        "sqft_lot",
        "sqft_basement",
        "sqft_living15",
        "sqft_lot15",
        "waterfront",
        "view",
        "distance",
    ],
    "target": "price",
    "prediction": "prediction",
    "datetime_features": ["year"],
}

from fastapi import FastAPI
import uvicorn
import mlflow
import os
import pandas as pd
from config import MODEL_S3_URI, MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

app = FastAPI(title="red wine quality prediction")

def get_chemical_attributes(product_id: int) -> pd.DataFrame:
    """
    Get chemical attributes of a wine based on its product_id. For simplicity,
    we just use hard-coded data here instead of fetching one from a real database. 
    """
    if product_id % 2 == 0:
        data = {"fixed acidity": [7.7], 
              "volatile acidity": [0.56], 
              "citric acid": [0.08], 
              "residual sugar": [2.5,], 
              "chlorides": [0.114], 
              "free sulfur dioxide": [14.0], 
              "total sulfur dioxide": [46.0], 
              "density": [0.9971], 
              "pH": [3.24], 
              "sulphates": [0.66], 
              "alcohol": [9.6]}
    else:
        data = {"fixed acidity": [6.7], 
              "volatile acidity": [0.46], 
              "citric acid": [0.24], 
              "residual sugar": [1.7], 
              "chlorides": [0.077], 
              "free sulfur dioxide": [18.0], 
              "total sulfur dioxide": [34.0], 
              "density": [0.9948], 
              "pH": [3.39], 
              "sulphates": [0.6], 
              "alcohol": [10.6]}
    return pd.DataFrame(data)

@app.on_event("startup")
def load_model():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    # Configure the credentials needed for accessing the MinIO storage service
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

    global model
    # load the model
    model = mlflow.pyfunc.load_model(model_uri=MODEL_S3_URI)

@app.get("/predict/{product_id}")
def predict(product_id: int):
    chemical_attrs = get_chemical_attributes(product_id)
    pred = model.predict(chemical_attrs)
    print(pred)
    return {"predicted score": pred[0]}

if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=3000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
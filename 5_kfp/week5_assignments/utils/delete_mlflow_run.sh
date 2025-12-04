#!/bin/bash
run_id=$1
mlflowpod=$(kubectl -n mlflow get pod -l app=mlflow | grep "mlflow-"|cut -d' ' -f1)
kubectl -n mlflow exec $mlflowpod -- mlflow gc --backend-store-uri postgresql://mlflow:KFSg-AYoiPdfRun64z2-w89Kk7z5cJL2IbVvSd3l8Og@postgres:5432/mlflowdb --run-ids $run_id
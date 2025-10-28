# Permanently delete MLflow experiments.
Even though you delete an MLflow experiment from the UI by clicking the rubbish bin button next to the experiment name, the experiment still exists in the PostgreSQL database used by MLflow. This will cause errors when you want to create an experiment or record runs under an experiment that has the same name as the experiment you deleted from the UI. To permanently delete an experiment, the following steps are required: 

Keep a note of the ID of the experiment you want to delete, then delete the experiment in the UI.

Shell into the pod where the MLflow server is running:
```bash
# first get the pod name
export mlflowpod="$(kubectl -n mlflow get pod -l app=mlflow | grep "mlflow-"|cut -d' ' -f1)"

# then shell into the pod
kubectl -n mlflow exec -it $mlflowpod -- bash
```
Now you should be inside the pod and see something like 
<img src="../images/shell-mlflow-pod.png">

Inside the pod, run the following command
```bash
mlflow gc --backend-store-uri postgresql://mlflow:KFSg-AYoiPdfRun64z2-w89Kk7z5cJL2IbVvSd3l8Og@postgres:5432/mlflowdb --experiment-ids <the-id-of-the-experiment-you-deleted-on-ui>
# The value given to --experiment-ids can be multiple experiment IDs separated by comma. If experiment-ids are not given, all experiments you deleted on the UI will be permanently deleted. For more info of mlflow gc, check https://mlflow.org/docs/2.9.2/cli.html?highlight=mlflow%20gc#mlflow-gc
```

If you have registered models corresponding to the MLflow runs under the deleted experiment, go back to the UI and delete them. You can delete all versions of a registered model:
<img src="../images/delete-all-versions-registered-model.png" />
or a single version of a registered model:
<img src="../images/delete-single-version-registered-model.png" />

#### How about permanently deleting MLflow runs?
Permanently deleting an MLflow run follows the similar steps: keep a note of the run ID -> delete the run from the UI -> shell into the mlflow pod -> run the following command
```bash
mlflow gc --backend-store-uri postgresql://mlflow:KFSg-AYoiPdfRun64z2-w89Kk7z5cJL2IbVvSd3l8Og@postgres:5432/mlflowdb --run-ids <your-deleted-run-id>
```
 -> clean the registered model version corresponding to the deleted run. 
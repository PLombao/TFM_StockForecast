import mlflow
import mlflow.pyfunc

def log_mlflow(experiment, model,
               params={}, metrics={}, tags={}):
    """
    Function to log models, params, metrics and tags to MLFlow
    """
    print("=== Logging in MLFlow Server...")
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        ## LOG PARAMS
        mlflow.log_params(params)
        print("Params logged")

        ## LOG METRICS
        mlflow.log_metrics(metrics)
        print("Metric logged.")

        ## LOG TAGS
        mlflow.set_tags(tags)
        print("Tags logged.")

        ## LOG MODEL
        if model != None:
            mlflow.pyfunc.log_model(artifact_path = "model",
                                    python_model = model,
                                    conda_env = "config/conda.yaml")
            runid = mlflow.active_run().info.run_uuid
            print("Model saved in run: {}.".format(runid))
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.helpers_mlflow import log_mlflow
from src.model import Model
from src.read_config import read_config_model
from src.prepare_data import split_data
from src.preprocess import Normalize, FeatureSelection
from src.validate import cross_validate
from sklearn.model_selection import train_test_split


def run(data, target, base_model, model_name):
    """
    Run a training of a machine learning model 
    Args:
        data (pd.DataFrame):        DataFrame with the filtered data
        target (str):               Name of the prediction target.              
        base_model (sklearn model): Scikit-learn object that will be used as base model for training.
      
    Returns:
        model:                      Trained model
        metrics (dict):             Dictionary with all the metrics for the model
    """    
    print('{:=^80}'.format('  RUN  '))
    
    # Split data
    train, test = train_test_split(data.reset_index(drop=True), test_size=0.1, shuffle=False)

    run_cv(train, target, base_model, model_name)

    train_x = train.drop([target], axis=1)
    test_x = test.drop([target], axis=1)
    train_y = train[[target]]
    test_y = test[[target]]

    # Init model
    norm = Normalize()
    pipeline = Pipeline([("norm", norm), ("model", base_model)])
    model = Model(pipeline)
    
    # Train model
    model.fit(train_x, train_y)

    # Evaluate model
    metrics = model.evaluate(test_x, test_y)
    metrics['features'] = train_x.shape[1]
    print(metrics)

    # Create df of predictions 
    pred_train = model._infer(train_x)
    predict_train = pd.DataFrame({"y_pred": pred_train, "y_real": np.array(train_y).reshape(-1)})
    predict_train["type"] = "train"
    pred_test = model._infer(test_x)
    predict_test = pd.DataFrame({"y_pred": pred_test, "y_real": np.array(test_y).reshape(-1)})
    predict_test["type"] = "test"
    predict = pd.concat([predict_train, predict_test])
    predict.index = data.index
    
    print('{:=^80}'.format(''))
    return model, metrics, predict

def run_cv(data, target, base_model, model_name, k=5):
    """
    Performing a CV training

    Args:
        data (pd.DataFrame):        DataFrame with the filtered data
        target (str):               Name of the prediction target.              
        base_model (sklearn model): Scikit-learn object that will be used as base model for training.
      
    Returns:
        model:                      Trained model
        metrics (dict):             Dictionary with all the metrics for the model
    """    
    experiment_name = "StockForecasting_1"

    # Get model info
    _, tags, params = read_config_model("stock", model_name)
    tags["modelo"] = model_name
    tags["target"] = target
    print('{:=^80}'.format('  RUN  '))
    print("Starting RUN on project {}.".format(experiment_name))

    # Init model
    norm = Normalize()
    pipeline = Pipeline([("norm", norm), ("model", base_model)])
    model = Model(pipeline)

    # Get CV metrics
    cv_metrics = cross_validate(model, data, target, k)
    print(cv_metrics)

    # Log in mlflow with no artifacts
    log_mlflow(experiment_name, None, params, cv_metrics, tags)

    return cv_metrics





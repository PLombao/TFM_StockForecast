import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# from src.helpers_mlflow import log_mlflow
from src.model import Model
from src.read_config import read_config_model
from src.prepare_data import split_data
from src.preprocess import Normalize, FeatureSelection
from src.validate import cross_validate

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
    # Set experiment name
    experiment_name = "StockForecasting"

    # Get model info
    _, tags, params = read_config_model("stock", model_name)

    print('{:=^80}'.format('  RUN  '))
    print("Starting RUN on project {}.".format(experiment_name))
    
    # Split data
    train_x, train_y, test_x, test_y = split_data(data, target, test_size=0.25)

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

    # LOG IN MLFLOW
    # log_mlflow(experiment_name, model, params, metrics, tags)

    print('{:=^80}'.format(''))
    return model, metrics

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
    experiment_name = "StockForecasting_CV"

    # Get model info
    _, tags, params = read_config_model("stock", model_name)

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
    # log_mlflow(experiment_name, None, params, cv_metrics, tags)

    return cv_metrics





import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.helpers_mlflow import log_mlflow
from src.model import Model
from src.read_config import read_config_model
from src.prepare_data import split_data
from src.preprocess import Normalize, FeatureSelection
from src.validate import cross_validate
from src.prepare_data import prepare_train_data
from sklearn.model_selection import train_test_split

def data_producto(data, prod):
    data = data.loc[data.producto == prod]
    data = prepare_train_data(data)

    data = data.loc[data.stockMissingType == 0].reset_index(drop=True)

    data = data[["fecha","producto","udsventa","udsprevisionempresa",
                 'udsstock']]

    # data = data[["fecha","producto","udsventa","udsprevisionempresa",
    #              "promo", "sin_weekday", "cos_weekday",
    #              "quarter", 'month','udsstock','udsprevisionempresa_shifted1', 'udsprevisionempresa_shifted2',
    #             'udsstock_diff7',"udsventa_diff1", "udsstock_shifted-1"]]

    return data

def run(data, target, base_model, params, tags):
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
    
    # 
    train_data = data.copy()
    train_data = train_data.drop(["producto","fecha"], axis=1)

    # Split data
    train, test = train_test_split(train_data.reset_index(drop=True), test_size=0.1, shuffle=False)

    run_cv(train, target, base_model, params, tags)

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
    predict = pd.concat([predict_train, predict_test]).reset_index(drop=True)
    predict["fecha"] = data["fecha"]
    predict["producto"] = data["producto"]
    
    print('{:=^80}'.format(''))
    return model, metrics, predict

def run_cv(data, target, base_model, params, tags, k=5):
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
    experiment_name = "StockForecasting_Enfoque"

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





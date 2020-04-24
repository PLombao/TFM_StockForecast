import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt
from random import shuffle

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt
def validate(actual, pred, y_mean=None):
    """
    Computes different metrics in order to validate a model (mape, mae, std_error, R2, RMSE). 
    
    Args:
        actual (list, np.array or a pd.Series): contains the values of target variable in the test set.
        pred (list, np.array or a pd.Series): contains the values of predicted target variable.
    Returns:
        metrics (dictionary):   a dictionary which cointains the metrics that will evaluate the prediction 
                                error of the model
    """
    metrics = {}
    actual, pred = np.array(actual), np.array(pred)
    # Base metrics
    metrics["mean_error"] = (actual-pred).mean()
    std_error = np.std(actual-pred)
    metrics["std_error"] = std_error
    metrics["r2"] = r2_score(actual,pred)
    metrics["predict_out"] = sum(pred>actual+2*std_error)+sum(pred<actual-2*std_error)

    # Scale dependant metrics
    metrics["mae"] = mean_absolute_error(actual,pred)
    metrics["rmse"] = sqrt(mean_squared_error(actual,pred))

    # Percentage metrics
    metrics["mape"] = np.mean(np.abs(actual - pred) / actual) * 100
    metrics["smape"] = np.mean(np.abs(actual - pred) / (actual + pred)) * 200

    # Relative error metrics (relative to avg prediction)
    if y_mean != None:
        metrics["mdrae"] = np.median(np.abs((actual-pred)/(actual-y_mean)))
    
    # Scale-free error metrics
    err_naive = np.mean(np.abs(np.diff(np.array(actual))))
    metrics["mase"] = np.mean((np.abs(actual-pred)/err_naive))
    
    return metrics


############### CROSS VALIDATION ######################################

def _kfold_cross_validation(items, k, randomize):
    """"
    A partir de un indice devuelve indices para CV
    Args:
        items (pandas index): 
        k (int): number of folds for cross validation
    Returns:
        metrics (dictionary):   a dictionary which cointains the metrics that will evaluate the prediction 
                                error of the model
    """
    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in range(k)]

    for i in range(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield training, validation

def _split_data_by_index(df, train_idx, test_idx, target):
    """"
    Split data by train and test index
    Args:
        df (pandas df):             data to cross validate
        train_idx/test_idx (list):  index of train and test to do the split
        target (str):               target variable to split on X/y
    Returns:
        X_train, X_test, y_train, y_test (pandas df)
    """
    # Preparamos los datos de entrada en los modelos
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    # Convertimos features y target en numpies
    X_train = df_train.drop(columns=[target], axis=1)
    X_test = df_test.drop(columns=[target], axis=1)
    y_train = df_train[target]
    y_test = df_test[target]

    return X_train, X_test, y_train, y_test

def cross_validate(model, df, target, k, randomize=False):
    """"
    Evaluate a model using MAPE and RMSE metrics and k-fold cross validation
    Args:
        model (python class):   untrained python model
        df (pandas df):         data to cross validate
        target (str):           name of the target variable
        k (int):                number of folds
        randomize (bool):       to shuffle or not the data
    Returns:
        metrics (dictionary):   a dictionary which cointains the metrics that will evaluate the prediction 
                                error of the model
    """
    print('{:=^80}'.format('  CROSS VALIDATE MODEL  '))
    # Initialize metrics list
    mape, rmse, mae, mean_error, std_error, smape, mdrae, mase = [], [], [], [], [], [], [], []
    print("Intializing Cross Validate Method...")
    # Iterate over the index for one fold
    for train_idx, test_idx in _kfold_cross_validation(df.index, k, randomize):
        # Split data for this fold
        X_train, X_test, y_train, y_test = _split_data_by_index(df, train_idx, test_idx, target)
        # Fit model
        try:
            model.fit(X_train, y_train)
            # Evaluate model
            try:
                metrics = model.evaluate(X_test, y_test)
                mape.append(metrics['mape'])
                rmse.append(metrics['rmse'])
                mae.append(metrics['mae'])
                mean_error.append(metrics['mean_error'])
                std_error.append(metrics['std_error'])
                smape.append(metrics['smape'])
                # mdrae.append(metrics['mdrae'])
                mase.append(metrics['mase'])
                
            except AttributeError:
                print("Model has not EVALUATE method...unable to evaluate")
                print("Only Custom Models with a evaluate method can be evaluated")
                break
        except AttributeError:
            print("model argument has not fit method...unable to FIT")
            break
    if mape != []:
        mape, rmse, mae, mean_error, std_error = np.array(mape), np.array(rmse), np.array(mae), np.array(mean_error), np.array(std_error)
        smape, mdrae, mase = np.array(smape), np.array(mdrae), np.array(mase), 
        results = {"mape_mean": mape.mean(), "mape_std":mape.std(),
                   "rmse_mean": rmse.mean(), "rmse_std":rmse.std(),
                   "mae_mean": mae.mean(), "mae_std":mae.std(),
                   "mean_error_mean": mean_error.mean(), "mean_error_std":mean_error.std(),
                   "std_error_mean": std_error.mean(), "std_error_std": std_error.std(),
                   "smape_mean":smape.mean(), "smape_std":smape.std(), 
                   # "mdrae_mean":mdrae.mean(), "mdrae_std":mdrae.std(),
                   "mase_mean":mase.mean(), "mase_std":mase.std()}
        print("Cross Validation done with {} folds".format(k))
        print('{:=^80}'.format(''))
        return results
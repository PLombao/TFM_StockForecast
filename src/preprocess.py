"""
FUNCTIONS UTILS TO PREPROCESS TRAIN
"""

import json
import pandas as pd
import numpy as np  
import sklearn
from sklearn.preprocessing import MinMaxScaler
from src.read_config import read_config_model

class FeatureSelection():

    def __init__(self, modelo, target="lab_density_avg"):
        self.modelo = modelo
        self.target = target

        # Obtener el diccionario con las variables del modelo y el num de features
        self.list_var_model, _, _ = read_config_model(self.modelo, self.target)
        self.features = len(self.list_var_model)

    def fit(self, x=None, y=None):
        return self

    def transform(self, x, y=None):
        """
        Select the variables which are used in a specific model and create a dataframe which 
        contains predictor variables.
        Args:
            df (pd.DataFrame):  a data frame which cointains all posible predictor variables.
            modelo (str): the name of the model (formato: espesor_anchoagrupado_material)
            target (str): two values are posible:  lab_density_avg, lab_ib_avg
        Returns:
            df_sel (pd.DataFrame):  a dataframe which contains the next columns: 
                                a datatime, a datakey, the target and predictor variables.
        """

        print('{:*^70}'.format('  FEATURE SELECTION  '))
        print("- Selecting variables (features) for material {} ...".format(self.modelo))
        # Filtrar el dataset de train por las variables que usa el modelo
        df_sel = x[self.list_var_model]
        print("Selected features for training. Data size: {}".format(self.features))
        print('{:*^70}'.format(''))
        return df_sel

    def fit_transform(self, x, y=None):
        self.fit()
        df = self.transform(x)
        return df

class Normalize():

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, x, y=None):
        self.scaler.fit(x.values)
    
    def transform(self, x, y=None):
        """
        Normalize to range (0,1) the predictor (numeric) variables of the dataset.
        Args:
            df (pd.DataFrame): a dataframe which cointains the predictor (numeric) variables of the dataset.
        Returns:
            df_norm (pd.DataFrame): a dataframe where the variables are normalized to range (0,1).
        """
        print('{:*^70}'.format('  NORMALIZE  '))
        print("- Normalizing predictor variables...")
        # Normalizacion min-max (solo variables type float)
        df_norm = pd.DataFrame(self.scaler.transform(x.values),columns=x.columns)
        print("Data normalized. Data size: {}".format(df_norm.shape))
        print('{:*^70}'.format(''))
        
        return df_norm
    
    def fit_transform(self, x, y=None):
        self.fit(x)
        df_nor = self.transform(x)
        return df_nor
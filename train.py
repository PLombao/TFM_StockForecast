import json
import pandas as pd
from src.load_data import load_data
from src.prepare_data import prepare_train_data
from src.trainer import run, run_cv


def train_monoproducto(data, base_model, modelo):
    prod = modelo.split("_")[1]
    data = data.loc[data.producto == prod]
    print(data.head())
    data = prepare_train_data(data)
    data = data[["udsventa","udsprevisionempresa", "udsstock"]]

    metrics = run_cv(data, "udsstock", base_model, modelo)

    return metrics



if __name__ == "__main__":

    # Load Stock data
    data = load_data()

    # Initalize base model
    from sklearn.linear_model import LinearRegression
    base_model = LinearRegression()
    
    arg1 = "all"
    
    # Si especificamos todos, entrena todos los modelos configurados
    if arg1 == "all":
        with open("config/model_stock.json") as config_file: 
            config = json.load(config_file)
        for modelo in list(config):
            metrics = train_monoproducto(data, base_model, modelo)
    else:
        print("DEMO MODE")
        print("")
        data = data.loc[data.producto == "1"]

        # Prepare data
        data = prepare_train_data(data)
        data = data[["udsventa", "udsstock", "uds"]]

        metrics = run_cv(data, "udsstock", base_model, "Model1")




        
        print(data.head())
        print(data.shape)
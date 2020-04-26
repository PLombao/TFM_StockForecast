import json
import pandas as pd
from src.load_data import load_data
from src.prepare_data import prepare_train_data
from src.trainer import run, run_cv


def data_producto(data, base_model, prod):
    data = data.loc[data.producto == prod]
    print(data.head())
    data = prepare_train_data(data)

    return data

def train(data, base_model, modelo, products):
    modeltype = modelo.split("_")[0]
    if modeltype == "PR":
        train_data = data_producto(data, base_model, products[0])
    elif modeltype == "CL":
        train_data = pd.DataFrame({})
        for prod in products:
            prod_data = data_producto(data, base_model, prod)
            train_data = pd.concat([train_data, prod_data])

    train_data = train_data.loc[train_data.working_day == 1].reset_index()
    train_data = train_data[["udsventa","udsprevisionempresa", "udsstock",
                 "promo", "sin_weekday", "cos_weekday",
                 "quarter", 'quarter', 'month','udsprevision_1', 'udsprevision_2',
                'udsprevision_7']]

    metrics = run_cv(train_data, "udsstock", base_model, modelo)

    return metrics

if __name__ == "__main__":

    # Load Stock data
    data = load_data()

    # # Initalize base model
    # from sklearn.linear_model import LinearRegression
    # base_model = LinearRegression()
    from sklearn.ensemble import RandomForestRegressor
    base_model = RandomForestRegressor(n_estimators=200)
    
    arg1 = "all"
    
    # Si especificamos todos, entrena todos los modelos configurados
    if arg1 == "all":
        with open("config/model_stock.json") as config_file: 
            config = json.load(config_file)
        for modelo in list(config):
            metrics = train(data, base_model, modelo, config[modelo]["productos"])
    else:
        print("DEMO MODE")
        print("")
        modelo = "PR_91"
        metrics = train(data, base_model, modelo, ["91"])
import json
import pandas as pd
from src.load_data import load_data

from src.trainer import data_producto, run, run_cv

def get_metadata(config_file, modelo):

    prods = config[modelo]["productos"]
    tags = {"model_name":modelo,
            "productos":" & ".join(prods)
            }
    return prods, tags

def train(data, base_model, modelo, products, params, tags):
    modeltype = modelo.split("_")[0]
    if modeltype == "PR":
        train_data = data_producto(data,products[0])
    elif (modeltype == "CL") | (modeltype == "ALL"):
        train_data = pd.DataFrame({})
        for prod in products:
            prod_data = data_producto(data, prod)
            train_data = pd.concat([train_data, prod_data])

    train_data = train_data.loc[train_data.working_day == 1].reset_index()
    train_data = train_data[["fecha","producto",'udsstock', "udsventa_shifted1", "udsstock_shifted1",
                            "udsprevisionempresa",
                            'roll4wd_udsprevisionempresa',
                             'working_day', 'month', "quarter","weekofyear",
                             'summer', 'autumn', 'winter',"cos_weekday","sin_weekday",
                              'udsprevisionempresa_shifted-1','udsprevisionempresa_shifted-6',
                               'udsstock_shifted7', 'roll4wd_udsstock_shifted7', 'roll4wd_udsstock'
                                ]]

    _, _, predict, metrics = run(train_data, "udsstock", base_model, params, tags)
    predict["modelo"] = modelo
    return metrics, predict

if __name__ == "__main__":

    # Load Stock data
    data = load_data()

    # # Initalize base model
    # from sklearn.linear_model import LinearRegression
    # base_model = LinearRegression()
    from sklearn.ensemble import RandomForestRegressor
    base_model = RandomForestRegressor(n_estimators=200)
    params = {"n_estimators":200}
    with open("config/model_stock.json") as config_file: 
        config = json.load(config_file)

    arg1 = "all"
    
    # Si especificamos todos, entrena todos los modelos configurados
    if arg1 == "all":
        
        metrics = []
        predicts = pd.DataFrame({})
        for modelo in list(config):
            prods, tags = get_metadata(config, modelo)
            metric, predict = train(data, base_model, modelo, prods, params, tags)
            metric["modelo"] = modelo
            metric["type"] = modelo.split("_")[0]
            metrics.append(metric)
            predicts = pd.concat([predicts, predict])

        pd.DataFrame(metrics).to_csv("metrics.csv", index=False)
        predicts.to_csv("predict.csv", index=False)
    elif arg1 == "demo":
        print("DEMO MODE")
        print("")
        modelo = "PR_91"
        prods, tags = get_metadata(config, modelo)
        metrics = train(data, base_model, modelo, prods, params, tags)
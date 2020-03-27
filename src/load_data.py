import pandas as pd
import numpy as np
from src.read_config import read_source_data
from src.utils import get_agg_stats
from src.cleaner import clean_ventas, clean_promos, clean_stock, clean_prevision

def load_csv(dataset):
    print(" - Loading dataset {}...".format(dataset))
    # Read dataset info
    [filename, dates, columns, shape] = read_source_data(dataset)

    # Load csv
    path = "data/" + filename
    df = pd.read_csv(path, sep=";", decimal = ",", encoding='latin-1',
                    parse_dates=dates, dtype=columns)

    # Correct columns for all lowcase
    df.columns = [col.lower() for col in df.columns]

    # Sort by date
    df = df.sort_values(dates[0])

    # Check shape of dataframe
    if df.shape != shape:
        print("[WARNING] Check out shape of dataset {}.".format(dataset))
        print("[WARNING] Should be {} and instead is {}.".format(shape, df.shape))

    # Clean datasets
    if dataset == "ventas": 
        df = clean_ventas(df)
    elif dataset == "promos":
        df = clean_promos(df)
    elif dataset == "stock":
        df = clean_stock(df)
    elif dataset == "prevision":
        df = clean_prevision(df)
    else:
        print("[WARNING] Cleaner not configured in load data for dataset {}.".format(dataset))


    print("     Dataset {} loaded.".format(dataset))
    return df

def load_ventas_byproduct(ventas=None):
    if type(ventas) == type(None):
        ventas = load_csv("ventas")
    print(" - Building dataset ventas by product")
    venta_stats, venta_names = get_agg_stats("venta")

    ventas_byprod = ventas.groupby("producto").agg({"fecha":["min","max","count"],
                                             "udsventa": venta_stats})
    ventas_byprod = ventas_byprod.reset_index()

    ventas_byprod.columns = ["producto", "fecha_primera_venta", "fecha_ultima_venta","freq_venta"] + venta_names
    print("    Dataset ventas by product builded")
    return ventas_byprod

def load_promos_range(promos=None):
    print(" - Building dataset promos by range")
    if type(promos) == type(None):
        promos = load_csv("promos")
    # Nos quedamos con las cols utiles y eliminamos promos repetidas
    promos = promos[["iniciopromo","finpromo","producto"]].drop_duplicates()

    # Inicializamos promos range
    promos_rng = pd.DataFrame({})

    for promo in promos.values:
        # Añadimos una promocion como rango
        dates = pd.date_range(promo[0], promo[1], freq='D')
        df_promo = pd.DataFrame({"fecha": dates,
                               "producto": promo[2]})
        promos_rng=pd.concat([promos_rng, df_promo])

    # Eliminamos promociones repetidas y añadimos flag de promo
    promos_rng = promos_rng.drop_duplicates()
    promos_rng["promo"] = 1

    print("    Dataset promos by range builded")  
    return promos_rng.reset_index(drop=True)

def load_data():
    """
    Loads data from src
    """
    ventas = load_csv("ventas")
    promos = load_csv("promos")
    stock = load_csv("stock")
    prevision = load_csv("prevision")
    festivos = load_csv("festivos")
    ventas_byprod = load_ventas_byproduct(ventas)
    promos_rng = load_promos_range(promos)
    
    return ventas, promos, stock, prevision, festivos, ventas_byprod, promos_rng

if __name__ == "__main__":
    print("TESTING LOAD DATA")
    ventas, promos, stock, prevision, festivos, \
        ventas_byprod,  promos_rng = load_data()



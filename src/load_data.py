import pandas as pd
import numpy as np
from src.read_config import read_source_data
from src.cleaner import clean_csv, clean_ventas, clean_promos, clean_stock, clean_prevision, clean_festivos
from src.cleaner import build_ventas_byproduct, build_promos_ranged, join_data

def load_csv(dataset):
    print('{:=^60}'.format('  LOAD DATASET {} '.format(dataset)))

    # Read dataset info
    [filename, dates, columns, shape] = read_source_data(dataset)

    # Load csv
    path = "data/raw/" + filename
    print("Reading CSV in {}...".format(path))
    df = pd.read_csv(path, sep=";", decimal = ",", encoding='latin-1',
                    parse_dates=dates, dtype=columns)

    # Check shape of dataframe file
    if df.shape != shape:
        print("[WARNING] Check out shape of dataset {}.".format(dataset))
        print("[WARNING] Should be {} and instead is {}.".format(shape, df.shape))

    # Clean csv
    df = clean_csv(df, dates[0])

    # Clean datasets
    if dataset == "ventas": 
        df = clean_ventas(df)
    elif dataset == "promos":
        df = clean_promos(df)
    elif dataset == "stock":
        df = clean_stock(df)
    elif dataset == "prevision":
        df = clean_prevision(df)
    elif dataset == "festivos":
        df = clean_festivos(df)
    else:
        print("[WARNING] Cleaner not configured in load data for dataset {}.".format(dataset))


    print("Dataset {} loaded. Shape: {}".format(dataset, df.shape))
    print('{:=^60}'.format(''))
    print("")
    return df.reset_index(drop=True)

def load_ventas_byproduct(ventas=None):
    if type(ventas) == type(None):
        ventas = load_csv("ventas")

    ventas_byprod = build_ventas_byproduct(ventas)
    return ventas_byprod

def load_promos_range(promos=None):
    if type(promos) == type(None):
        promos = load_csv("promos")
    
    promos_rng = build_promos_ranged(promos)
    return promos_rng

def load_data():
    """
    Loads data from src
    """
    stock = load_csv("stock")
    ventas = load_csv("ventas")
    prevision = load_csv("prevision")
    festivos = load_csv("festivos")
    promos_rng = load_promos_range()

    stock = join_data(stock, ventas, prevision, promos_rng, festivos)

    return stock

if __name__ == "__main__":
    print("TESTING LOAD DATA")

    stock = load_data()

    # # print(stock.info())
    print(stock.head())


    # print("Duplicates")
    # print(stock.shape)
    # print(stock.drop_duplicates().shape)



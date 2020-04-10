import pandas as pd
import numpy as np
from src.read_config import read_source_data
from src.cleaner_datasets import clean_ventas, clean_promos, clean_stock, clean_prevision, clean_festivos
from src.cleaner_utils import clean_csv
from src.builder import build_ventas_byproduct, build_stock_byproduct, build_promos_ranged
from src.cleaner import join_data

def load_raw_csv(dataset):
    # Read dataset info
    [filename, dates, columns, shape] = read_source_data(dataset)

    # Load csv
    path = "data/raw/" + filename
    print("Reading CSV in {}...".format(path))
    df = pd.read_csv(path, sep=";", decimal = ",", encoding='latin-1',
                    parse_dates=dates, dtype=columns, dayfirst=True)

    # Check shape of dataframe file
    if df.shape != shape:
        print("[WARNING] Shape should be {} and instead is {}.".format(shape, df.shape))

    # Clean csv
    df = clean_csv(df, dates[0])

    return df

def load_csv(dataset):
    print('{:=^60}'.format('  LOAD DATASET {} '.format(dataset)))

    # Load raw csv
    df = load_raw_csv(dataset)

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

def load_stock_byproduct(stock=None):
    if type(stock) == type(None):
        stock = load_csv("stock")

    stock_byprod = build_stock_byproduct(stock)
    return stock_byprod

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

def load_clustering_data():
    """
    Loads the data for the clustering (with zeros instead of nans)
    """
    # Cargamos los datos de stock para tener todas las fechas
    data = load_data()
    data = data[['fecha','producto', 'udsventa']]

    # Asignamos nulls como 0 en uds venta (TEMPORAL)
    data.udsventa = data.udsventa.fillna(0)
    print("Assingning nulls in udsVenta as 0 [TEMPORARY]")

    # Quitamos los dias 23 a 26
    data = data.loc[data.fecha < '2020-03-23']
    print("Dropping dates from 23 to 26 March 20")

    clustering = build_ventas_byproduct(data)

    return clustering

if __name__ == "__main__":
    print("TESTING LOAD DATA")

    # df = load_csv("ventas")
    # df = load_csv("prevision")
    # stock = load_csv("stock")
    # ventas = load_csv("ventas")
    # prevision = load_csv("prevision")
    # festivos = load_csv("festivos")
    # promos_rng = load_promos_range()
    # print(df.info())
    # print(df.head())
    # print(df.shape)


    print(load_clustering_data())











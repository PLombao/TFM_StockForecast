import pandas as pd
from src.read_config import read_source_data
from src.utils import get_agg_stats

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

    print("     Dataset {} loaded.".format(dataset))
    return df

def load_ventas_byproduct(ventas=None):
    if type(ventas) == None:
        ventas = load_csv("ventas")
    print(" - Building dataset ventas by product")
    venta_stats, venta_names = get_agg_stats("venta")

    ventas_byprod = ventas.groupby("producto").agg({"fecha":["min","max","count"],
                                             "udsventa": venta_stats})
    ventas_byprod = ventas_byprod.reset_index()

    ventas_byprod.columns = ["producto", "fecha_primera_venta", "fecha_ultima_venta","freq_venta"] + venta_names
    print("    Dataset ventas by product builded")
    return ventas_byprod

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
    
    return ventas, promos, stock, prevision, festivos, ventas_byprod


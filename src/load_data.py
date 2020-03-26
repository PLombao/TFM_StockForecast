import pandas as pd
from src.read_config import read_source_data

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

    # Check shape of dataframe
    if df.shape != shape:
        print("[WARNING] Check out shape of dataset {}.".format(dataset))
        print("[WARNING] Should be {} and instead is {}.".format(shape, df.shape))

    print("     Dataset {} loaded.".format(dataset))
    return df

def load_data():
    """
    Loads data from src
    """
    ventas = load_csv("ventas")
    promos = load_csv("promos")
    stock = load_csv("stock")
    prevision = load_csv("prevision")
    festivos = load_csv("festivos")
    
    return ventas, promos, stock, prevision, festivos


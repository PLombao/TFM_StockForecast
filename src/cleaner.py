import numpy as np
import pandas as pd

def join_data(stock, ventas, prevision, promos_rng, festivos):
    print('{:=^60}'.format('  JOIN DATASET STOCK  '))
    print("Input shape: {}".format(stock.shape))

    # TEMPORARY: grouping stock by product and date
    stock = stock.groupby(["fecha","producto"]).agg(lambda x: int(x.mean())).reset_index()
    print("[WARNING] Dropping duplicates in fecha | producto for stock data [TEMPORARY]")

    fecha_ini = stock.fecha.min().date()
    fecha_fin = stock.fecha.max().date()
    print("Stock dataframe received with data from {} to {}".format(fecha_ini, fecha_fin))
    
    # Creamos un df con todas las combinaciones de fechas y productos del dataset stock
    total_dates = pd.DataFrame({"fecha": pd.date_range(fecha_ini, fecha_fin, freq='D'),
                                "key":1})
    total_products = pd.DataFrame({"producto": stock.producto.sort_values().unique(),
                                   "key":1})
    total = total_dates.merge(total_products, on='key').drop("key", axis=1)
    print("Created dataframe for {} days and {} products [shape: {}]"\
        .format(total_dates.shape[0],total_products.shape[0], total.shape))

    # Hacemos un merge con los datos de stock, ventas, prevision y promos range
    for df, name in zip([stock, ventas, prevision, promos_rng], ['stock', 'ventas','prevision', "promos range"]):
        total = total.merge(df, on=['fecha', 'producto'], how='left')
        print("Merged dataframe with {} data    [shape: {}]".format(name, total.shape))

    # Hacemos un merge con los datos de festivos
    total = total.merge(festivos, on=['fecha'], how='left')
    print("Merged dataframe with festivos data    [shape: {}]".format(total.shape))

    # Assign missings to 0 in promo & festivo
    total["promo"] = total["promo"].fillna(0)
    total["festivo"] = total["festivo"].fillna(0)

    # Add new columns weekday
    total['weekday'] = total['fecha'].apply(lambda x: x.weekday())

    print("Output shape: {}".format(total.shape))
    print('{:=^60}'.format(''))
    print("")
    return total
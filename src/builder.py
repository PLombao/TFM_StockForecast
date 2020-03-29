import numpy as np
import pandas as pd

from src.utils import get_agg_stats

def build_ventas_byproduct(ventas):
    print('{:=^60}'.format('  BUILD RANGED PROMOS  '))
    venta_stats, venta_names = get_agg_stats("venta")

    ventas_byprod = ventas.groupby("producto").agg({"fecha":["min","max","count"],
                                             "udsventa": venta_stats})
    ventas_byprod = ventas_byprod.reset_index()

    ventas_byprod.columns = ["producto", "fecha_primera_venta", "fecha_ultima_venta","freq_venta"] + venta_names
    print("Dataset ventas by product builded")
    print('{:=^60}'.format(''))
    return ventas_byprod

def build_promos_ranged(promos):
    print('{:=^60}'.format('  BUILD RANGED PROMOS  '))
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

    print("Dataset promos from intervals to range builded.")
    print('{:=^60}'.format(''))
    print("")
    return promos_rng.reset_index(drop=True)

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
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
    # TEMPORARY: grouping stock by product and date
    stock = stock.groupby(["fecha","producto"]).agg(lambda x: int(x.mean())).reset_index()
    print("[WARNING] Dropping duplicates in fecha | producto for stock data [TEMPORARY]")
    print("Input shape: {}".format(stock.shape))

    # Convert date to str for the join
    stock["idfecha"] = stock.fecha.apply(lambda x: x.strftime('%Y%m%d'))

    # Joining date | product
    for df in [ventas, prevision, promos_rng]:
        df["idfecha"] = df.fecha.apply(lambda x: x.strftime('%Y%m%d'))
        df = df.drop("fecha", axis=1)
        stock = stock.merge(df, on=['idfecha', 'producto'], how='left')
    
    # Joining date
    for df in [festivos]:
        df["idfecha"] = df.fecha.apply(lambda x: x.strftime('%Y%m%d'))
        df = df.drop("fecha", axis=1)
        stock = stock.merge(df, on=['idfecha'], how='left')

    # Assign missings to 0 in promo & festivo
    stock["promo"] = stock["promo"].fillna(0)
    stock["festivo"] = stock["festivo"].fillna(0)

    # Drop id fecha
    stock = stock.drop("idfecha", axis=1)
    
    # stock = stock.drop("idfecha")
    print("Output shape: {}".format(stock.shape))
    print('{:=^60}'.format(''))
    return stock
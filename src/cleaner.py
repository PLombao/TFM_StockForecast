import numpy as np
import pandas as pd

from src.utils import get_agg_stats

def clean_csv(df, datecol):
    print('{:=^40}'.format('  CLEAN CSV  '.format()))
    # Correct columns for all lowcase
    df.columns = [col.lower() for col in df.columns]
    print("Set columns to lowcase.")

    # Sort by date
    df = df.sort_values(datecol)
    print("Sort values by date.")

    # Drop duplicates
    filter_data = df.drop_duplicates()
    print("Dropped duplicates. Rows dropped: {}."\
        .format(df.shape[0]-filter_data.shape[0]))

    print('{:=^40}'.format(''.format()))
    return filter_data

def clean_ventas(data):
    print('{:=^40}'.format('  CLEAN VENTAS  '.format()))

    if data.shape[0] != data[['fecha',"producto"]].drop_duplicates().shape[0]:
        print("[WARNING] Ventas data with different units for same product & data. Rows: {}"\
            .format(data.shape[0] - data[['fecha',"producto"]].drop_duplicates().shape[0]))

    # Pasamos a enteros las unidades truncando
    data.udsventa = data.udsventa.apply(lambda x: int(x))
    print("UnidadesVentas to integer.")

    # Eliminamos ceros (no se especifican todos los ceros)
    filter_data = data.loc[data.udsventa != 0].reset_index(drop=True)
    print("Drop rows with zero in UnidadesVenta.")
    print("  Rows dropped: {}"\
        .format(data.shape[0]-filter_data.shape[0]))

    print('{:=^40}'.format(''.format()))
    return filter_data

def clean_promos(data):
    print('{:=^40}'.format('  CLEAN PROMOS  '.format()))
    # Eliminamos promociones sin fecha final
    filter_data = data.loc[~data.finpromo.isna()].reset_index(drop=True)
    print("Drop rows of promos without end date:")
    print("   Rows dropped: {}"\
        .format(data.shape[0]-filter_data.shape[0]))
    print("   The most recent of the promos dropped was started in {}"\
        .format(data.loc[data.finpromo.isna(), "iniciopromo"].max()))
    print('{:=^40}'.format(''.format()))
    return filter_data

def clean_stock(data):
    print('{:=^40}'.format('  CLEAN STOCK  '.format()))

    if data.shape[0] != data[['fecha',"producto"]].drop_duplicates().shape[0]:
        print("[WARNING] Stock data with different units for same product & data. Rows: {}"\
            .format(data.shape[0] - data[['fecha',"producto"]].drop_duplicates().shape[0]))

    print('{:=^40}'.format(''.format()))
    return data

def clean_prevision(data):
    print('{:=^40}'.format('  CLEAN PREVISION  '.format()))
    if data.shape[0] != data[['fecha',"producto"]].drop_duplicates().shape[0]:
        print("[WARNING] Ventas data with different units for same product & data. Rows: {}"\
            .format(data.shape[0] - data[['fecha',"producto"]].drop_duplicates().shape[0]))

    data.udsprevisionempresa = data.udsprevisionempresa.apply(lambda x: int(x))
    print("UnidadesPrevisionVentas to integer.")
    print('{:=^40}'.format(''.format()))
    return data

def clean_festivos(data):
    print('{:=^40}'.format('  CLEAN FESTIVOS  '.format()))
    data = data.loc[:,['fecha', 'festivo']]
    data.festivo = data.festivo.apply(lambda x: 1)
    print("Drop unnecesary columns.")
    print('{:=^40}'.format(''.format()))

    return data

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
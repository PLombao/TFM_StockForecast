import numpy as np
import pandas as pd
from src.cleaner_utils import check_len_ts, clean_csv

def clean_ventas(data):
    print('{:=^40}'.format('  CLEAN VENTAS  '.format()))

    if data.shape[0] != data[['fecha',"producto"]].drop_duplicates().shape[0]:
        print("[WARNING] Ventas data with different units for same product & data. Rows: {}"\
            .format(data.shape[0] - data[['fecha',"producto"]].drop_duplicates().shape[0]))

    check_len_ts(data, "fecha")

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
    
    # Calculamos el ahorro de la promo
    filter_data['ahorro'] = filter_data['preciotarifa'] - filter_data['preciopromocion']
    print("Created new variable ahorro (preciotarifa - preciopromo)")
    print('{:=^40}'.format(''.format()))
    return filter_data

def clean_stock(data):
    print('{:=^40}'.format('  CLEAN STOCK  '.format()))

    check_len_ts(data, "fecha")

    if data.shape[0] != data[['fecha',"producto"]].drop_duplicates().shape[0]:
        print("[WARNING] Stock data with different units for same product & data. Rows: {}"\
            .format(data.shape[0] - data[['fecha',"producto"]].drop_duplicates().shape[0]))

    print('{:=^40}'.format(''.format()))
    return data

def clean_prevision(data):
    print('{:=^40}'.format('  CLEAN PREVISION  '.format()))

    check_len_ts(data, "fecha")
    
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


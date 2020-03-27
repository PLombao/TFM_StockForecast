
def clean_ventas(data):
    print("DATA CLEANING VENTAS")

    # Pasamos a enteros las unidades truncando
    data.udsventa = data.udsventa.apply(lambda x: int(x))
    print(" - Pasadas unidades venta a enteros.")

    # Eliminamos ceros (no se especifican todos los ceros)
    filter_data = data.loc[data.udsventa != 0].reset_index(drop=True)
    print(" - Eliminados registros con cero en las unidades venta. Registros eliminados: {}"\
        .format(data.shape[0]-filter_data.shape[0]))

    return filter_data

def clean_promos(data):
    # Eliminamos promociones sin fecha final
    filter_data = data.loc[~data.finpromo.isna()].reset_index(drop=True)
    print(" - Eliminados registros con null en fecha final de promocion. Registros eliminados: {}"\
        .format(data.shape[0]-filter_data.shape[0]))
    print("   La fecha más reciente de inicio de promocion de los eliminados es {}"\
        .format(data.loc[data.finpromo.isna(), "iniciopromo"].max()))

    return filter_data

def clean_stock(data):
    return data

def clean_prevision(data):
    data.udsprevisionempresa = data.udsprevisionempresa.apply(lambda x: int(x))
    print(" - Pasadas unidades prevision de venta a enteros.")

    return data

def clean_data(data):

    return data
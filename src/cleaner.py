
def clean_ventas(data):
    print("DATA CLEANING VENTAS")

    # Pasamos a enteros las unidades truncando
    data.udsventa = data.udsventa.apply(lambda x: int(x))
    print(" - Pasadas unidades venta a enteros.")

    # Eliminamos ceros (no se especifican todos los ceros)
    filter_data = data.loc[data.udsventa != 0]
    print(" - Eliminados registro con cero en las unidades venta. Registros eliminados: {}"\
        .format(data.shape[0]-filter_data.shape[0]))

    return data

def clean_promos(data):
    return data

def clean_stock(data):
    return data

def clean_prevision(data):
    data.udsprevisionempresa = data.udsprevisionempresa.apply(lambda x: int(x))
    print(" - Pasadas unidades prevision de venta a enteros.")
    
    return data

def clean_data(data):

    return data
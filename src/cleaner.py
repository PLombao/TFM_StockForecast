

def clean_ventas(data):
    data.udsventa = data.udsventa.apply(lambda x: trunc(x))
    return data

def clean_promos(data):
    return data

def clean_stock(data):
    return data

def clean_prevision(data):
    return data

def clean_data(data):

    return data
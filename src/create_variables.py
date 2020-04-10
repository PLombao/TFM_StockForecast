import pandas as pd

def get_time_variables(data, datecol):
    # Add new columns weekday
    data['weekday'] = data[datecol].apply(lambda x: x.weekday())

    return data

def get_stockMissingTypeByProd(ts):
    """
    Creates a variable of the type of missings:
    0 - is not a missing
    1 - is a missing of a 1 day interval
    2 - is a missing of a > 1 day interval
    """
    # Get missings in stock
    ts['missing'] = (ts.udsstock.isna()) * 1
    
    # From those missings, get what are from intervals > 1 day
    subts = ts.loc[ts.missing == 1,["fecha"]]
    # Getting the diff backward and forward
    subts['fw'] = subts.fecha.diff(1).apply(lambda x: x.days)
    subts['bw'] = subts.fecha.diff(-1).apply(lambda x: x.days)
    # Si la diferencia en dias con el anterior y el siguiente es mayor de 1 en ambas, el intervalo es de 1 dia
    subts['interval'] = (~((subts.fw > 1) & (subts.bw < -1))) * 1
    subts = subts.drop(['fw','bw'], axis=1)
    
    # Juntamos los datos y asginamos 0s a los intervals nulls
    ts = ts.merge(subts, on='fecha', how='left')
    ts.interval = ts.interval.fillna(0)
    
    # Creamos variable stockMissingType
    ts['stockMissingType'] = ts['missing'] + ts['interval']
    ts = ts.drop(['missing','interval'], axis=1)
    return ts

def get_stockMissingType(df):
    new_df = pd.DataFrame({})
    for product in df.producto.unique():
        new_df = pd.concat([new_df, get_stockMissingTypeByProd(df.loc[(df.producto==product)])])
    return new_df


def create_variables(df):
    df = get_stockMissingType(df)
    df = get_time_variables(df, "fecha")

    return df


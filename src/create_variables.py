import pandas as pd
import numpy as np

def get_time_variables(df, datecol):
    print("Getting datetime variables: weekday, quarter, month, weekofyear")
    # Add new columns weekday
    df['weekday'] = df[datecol].dt.dayofweek
    df['quarter'] = df[datecol].dt.quarter
    df['month'] = df[datecol].dt.month
    df['weekofyear'] = df[datecol].dt.weekofyear
    df['working_day'] = (df['weekday'] != 6) & (df['festivo'] == 0)
    df['sin_weekday'] = df['weekday'].apply(lambda x: np.sin((2*np.pi/7)*x))
    df['cos_weekday'] = df['weekday'].apply(lambda x: np.cos((2*np.pi/7)*x))

    df['is_august'] = (df['month'] == 8) * 1
    df["spring"] = ((df[datecol]>'2020-03-21') & (df[datecol]<'2020-06-21')) * 1
    df["summer"] = ((df[datecol]>'2019-06-01') & (df[datecol]<'2019-09-21')) * 1
    df["autumn"] = ((df[datecol]>'2019-09-21') & (df[datecol]<'2019-12-21')) * 1
    df['winter'] = (df[datecol]>'2019-12-21') * 1
    return df

def _get_stockMissingTypeByProd(ts):
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
    print("Get stock missing type")
    return ts

def _get_roll4wd(day_data, col):
    day_data["roll4wd_" + col] = day_data.loc[(day_data.festivo == 0), col].rolling(4, win_type='triang', min_periods=1).mean()
    day_data["meanwd_" + col] = day_data.loc[(day_data.festivo == 0), col].mean()
    print("Getting rolling windows of last 5 days by product and weekday for column {}".format(col))
    return day_data

def get_dateproduct_variables(prod_data):
    drop_cols = prod_data.loc[:,~prod_data.columns.isin(["fecha", "producto"])].columns
    out_prod_data = prod_data.copy()
    out = pd.DataFrame({})
    for wd in prod_data['weekday'].unique():
        day_data = prod_data.loc[(prod_data.weekday == wd)]
        # Get roll4wd and meanwd for stock, venta and prevision
        for col in ['udsventa', 'udsstock', 'udsprevisionempresa']:
            day_data = _get_roll4wd(day_data, col)
        out = pd.concat([out, day_data])
    out_prod_data = prod_data.drop(drop_cols, axis=1)
    out_prod_data = out_prod_data.merge(out, how='left', on=['fecha','producto'])
    return out_prod_data
    
def get_product_variables(data):
    # Columnas a eliminar antes de hacer el merge
    drop_cols = data.loc[:,~data.columns.isin(["fecha", "producto"])].columns
    out_data = data.copy()
    # Creamos un dataframe vacio
    out = pd.DataFrame({})
    for product in data['producto'].unique():
        prod_data = data.loc[(data['producto'] == product)]

        # Creamos la variable de tipo stock missing
        prod_data = _get_stockMissingTypeByProd(prod_data)
        # Get variables of date prod
        prod_data = get_dateproduct_variables(prod_data)
        
        out = pd.concat([out, prod_data])
    out_data = data.drop(drop_cols, axis=1)
    out_data = out_data.merge(out, how='left', on=['fecha','producto'])
    return out_data

def create_variables(df):
    print('{:=^40}'.format('  CREATE VARIABLES  '))
    df = get_time_variables(df, "fecha")
    df = get_product_variables(df)
    print('{:=^40}'.format(''))
    return df


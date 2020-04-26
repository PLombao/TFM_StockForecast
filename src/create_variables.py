import pandas as pd
import numpy as np

def get_time_variables(df, datecol):
    print("Getting datetime variables: weekday, quarter, month, weekofyear")
    # Add new columns weekday
    df['weekday'] = df[datecol].dt.dayofweek
    df['quarter'] = df[datecol].dt.quarter
    df['month'] = df[datecol].dt.month
    df['weekofyear'] = df[datecol].dt.weekofyear
    df['working_day'] = (df['weekday'] == 6) | (df['festivo'] == 1)
    df['sin_weekday'] = df['weekday'].apply(lambda x: np.sin((2*np.pi/7)*x))
    df['cos_weekday'] = df['weekday'].apply(lambda x: np.cos((2*np.pi/7)*x))

    return df

def get_shifted_prevision(data, period):
    out = pd.DataFrame({})
    for product in data['producto'].unique():
        prod_data = data.loc[(data['producto'] == product)]
        prod_data['udsprevision_' + str(period)] = data['udsprevisionempresa'].shift(periods=period, fill_value=0)
        out = pd.concat([out, prod_data])
    data = data.merge(out[['fecha','producto', 'udsprevision_' + str(period)]], how='left', on=['fecha','producto'])
    print("Created new variable with the prevision ventas shifted {} period/s.".format(period))
    return data

def get_diff_variable()

def get_roll4wd(data, col):
    print("Getting rolling windows of last 5 days by product and weekday for column {}".format(col))
    colname = 'roll4wd_' + col
    out = pd.DataFrame({})
    for product in data['producto'].unique():
        prod_data = data.loc[(data['producto'] == product)]
        for wd in data['weekday'].unique():
            day_data = prod_data.loc[(prod_data.festivo == 0) & (prod_data.weekday == wd)]
            day_data[colname] = day_data[col].rolling(4, win_type='triang', min_periods=1).mean()
            day_data["meanwd_"+col] = day_data[col].mean()
            out = pd.concat([out, day_data])
            
    data = data.merge(out[['fecha','producto', colname, "meanwd_"+col]], how='left', on=['fecha','producto'])
    return data

def get_deltaStock(data):
    print("Getting deltaStock as the difference of Stock from today to tomorrow")
    out = pd.DataFrame({})
    for product in data['producto'].unique():
        prod_data = data.loc[(data['producto'] == product)]
        prod_data['deltaStock'] = prod_data.udsstock.diff(periods=-1)
        out = pd.concat([out, prod_data])
    data = data.merge(out[['fecha','producto', 'deltaStock']], how='left', on=['fecha','producto'])
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
    print("Getting stock missing type")
    new_df = pd.DataFrame({})
    for product in df.producto.unique():
        new_df = pd.concat([new_df, get_stockMissingTypeByProd(df.loc[(df.producto==product)])])
    df = df.merge(new_df[['fecha','producto','stockMissingType']], how='left', on=['fecha','producto'] )
    return df

def create_variables(df):
    df = get_stockMissingType(df)
    df = get_time_variables(df, "fecha")
    df = get_deltaStock(df)
    for col in ['udsventa', 'udsstock', 'udsprevisionempresa']:
        df = get_roll4wd(df, col)
    
    # Create the prevision ventas shifted for the seventh first days
    for period in range(1,8):
        df = get_shifted_prevision(df, period)

    return df


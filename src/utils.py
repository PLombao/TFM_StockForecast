from scipy.stats import iqr, skew, kurtosis
import numpy as np
import pandas as pd

def get_agg_stats(variable):
    agg_stats = ["min", lambda x: np.quantile(x,q=.05), 
                lambda x: np.quantile(x,q=.25), "median",
                lambda x: np.quantile(x,q=.75),
                lambda x: np.quantile(x,q=.95) ,
                "max", iqr, "mean", "std", lambda x: np.std(x)/np.mean(x),
                skew, kurtosis, "mad"]
    agg_names = ["min", "q5","q25","median","q75","q95",
                "max", "iqr", "mean", "std", "CV",
                "skew","kurtosis", "mad"]
    agg_names = [variable + "_" + x for x in agg_names]
    return agg_stats, agg_names

def get_dateagg_stats():
    """
    Calculates the min_date, max_date, count and intermitence for a date column
    """
    dateagg_stats = ["min","max","count",
                    lambda x: x.diff().mean().total_seconds()]
    dateagg_names = ["fecha_min", "fecha_max","frecuencia","intermitencia"]

    return dateagg_stats, dateagg_names


def get_ratios(df):
    datecol = "fecha"
    df['weekday'] = df[datecol].dt.dayofweek
    df['month'] = df[datecol].dt.month
    out = []
    for prod in df.producto.unique():
        outdict = {}
        outdict['producto'] = prod
        outdict['aug_ratio'] = df.loc[(df.producto == prod) & (df.month == 8),"udsventa"].mean()/df.loc[(df.producto == prod),"udsventa"].mean()
        for wd in df['weekday'].unique():
            outdict['wd_ratio_'+str(wd+1)] = df.loc[(df.producto == prod) & (df.weekday == wd),"udsventa"].mean()/df.loc[(df.producto == prod),"udsventa"].mean()
        
        out.append(outdict)
    
    return pd.DataFrame(out)

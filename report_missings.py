import pandas as pd

# PRINT EDA FOR STOCK DATA
df = pd.read_csv("data/clean/stock_all.csv")

def filter_festivos(df, only):
    if only:
        df = df.loc[(df.festivo == 1) & (df.weekday == 6)]
    else: 
        df = df.loc[(df.festivo != 1) & (df.weekday != 6)]
    return df
def perc(df1, df2):
    return round(100* df1.shape[0]/df2.shape[0])

def missing_analysis(variable):

    exist = df.loc[~df[variable].isna()]
    miss = df.loc[df[variable].isna()]
    print("  Valores totales: {} ".format(df.shape[0]))
    print("  Valores existentes en dias de vacaciones: {}"\
                .format(filter_festivos(exist, True).shape[0]))
    print("  Valores missings in udsventa {} ({}%)"\
                .format(miss.shape[0], perc(miss, df)))
    miss_noholiday = filter_festivos(miss, False)
    print("  Valores missings in udsventa sin domingos y festivos {} ({}%)"\
                .format(miss_noholiday.shape[0], perc(miss_noholiday, df)))
    
    byprod = miss_noholiday.groupby("producto")['fecha'].count().reset_index().sort_values("fecha", ascending=False)
    byprod.columns = ["producto", "num_missings"]
    print("   Missings en udsventa sin domingos y festivos para {} productos de {} totales."\
                .format(byprod.loc[byprod.num_missings != 0].shape[0], byprod.shape[0]))
    print("   5 productos con más missings:")
    print(byprod.head())

    bydate = miss_noholiday.groupby("fecha")['producto'].count().reset_index().sort_values("producto", ascending=False)
    bydate.columns = ["fecha", "num_missings"]
    print("   Missings en udsventa sin domingos y festivos para {} fechas de {} totales."\
                .format(bydate.loc[bydate.num_missings != 0].shape[0], bydate.shape[0]))
    print("   5 fechas con más missings:")
    print(bydate.head())
    
    

print("MISSINGS FOR STOCK ALL DATA")
print("")
print("*"*60)
print("MISSINGS in ventas")
missing_analysis("udsventa")










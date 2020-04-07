import pandas as pd
from src.load_data import load_csv, load_ventas_byproduct, load_promos_range, load_stock_byproduct, load_data, load_clustering_data


profiling = False
if profiling:
    from pandas_profiling import ProfileReport

# PRINT EDA FOR THE ORIGINAL CSV  
for name in ["ventas", "promos", "stock","prevision", "festivos"]:

    df = load_csv(name)
    
    df.to_csv("data/clean/"+ name + ".csv", index=False)
    if profiling:
        if name=="promos": df = df.drop(['semanafin', 'cantidad'], axis = 1)
        profile = ProfileReport(df, title='EDA for {} dataset.'.format(name))
        profile.to_file(output_file="reports/eda/"+name+".html")

# PRINT EDA FOR VENTAS BY PROD (CLUSTERING DATA)
ventas_byprod = load_ventas_byproduct()
ventas_byprod.to_csv("data/clean/ventas_byprod.csv", index=False)
if profiling:
    profile = ProfileReport(df, title='EDA for ventas by prod dataset.')
    profile.to_file(output_file="reports/eda/ventas_byprod.html")

# PRINT EDA FOR PROMOS IN RANGE DATES
promos_range = load_promos_range()
promos_range.to_csv("data/clean/promos_range.csv", index=False)
if profiling:
    profile = ProfileReport(promos_range, title='EDA for promos by range dataset.')
    profile.to_file(output_file="reports/eda/promos_byrange.html")

# PRINT EDA FOR STOCK BY PROD
stock_byprod = load_stock_byproduct()
stock_byprod.to_csv("data/clean/stock_byprod.csv", index=False)
if profiling:
    profile = ProfileReport(df, title='EDA for stock by prod dataset.')
    profile.to_file(output_file="reports/eda/stock_byprod.html")

# PRINT EDA FOR CLUSTERING DATA
clustering = load_clustering_data()
clustering.to_csv("data/clean/clustering.csv")
if profiling:
    profile = ProfileReport(df, title='EDA for STOCK dataset.')
    profile.to_file(output_file="reports/eda/stock_all.html")

# PRINT EDA FOR STOCK DATA
df = load_data()
df.to_csv("data/clean/stock_all.csv", index=False)
if profiling:
    profile = ProfileReport(df, title='EDA for STOCK dataset.')
    profile.to_file(output_file="reports/eda/stock_all.html")


print("MISSINGS FOR STOCK ALL DATA")
miss_stock = df.loc[df.udsstock.isna()]
print("{} missings in udsstock of total rows {}".format(miss_stock.shape[0], df.shape[0]))
miss_stock_wd = miss_stock.loc[(miss_stock.festivo != 1) & (miss_stock.weekday != 6)]
print("{} missings in udsstock without holidays".format(miss_stock_wd.shape[0]))

miss_stock = df.loc[df.udsventa.isna()]
print("{} missings in udsventa of total rows {}".format(miss_stock.shape[0], df.shape[0]))
miss_stock_wd = miss_stock.loc[(miss_stock.festivo != 1) & (miss_stock.weekday != 6)]
print("{} missings in udsventa without holidays".format(miss_stock_wd.shape[0]))


miss_stock = df.loc[df.udsprevisionempresa.isna()]
print("{} missings in udsprevisionventa of total rows {}".format(miss_stock.shape[0], df.shape[0]))
miss_stock_wd = miss_stock.loc[(miss_stock.festivo != 1) & (miss_stock.weekday != 6)]
print("{} missings in udsprevisionventa without holidays".format(miss_stock_wd.shape[0]))


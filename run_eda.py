import pandas as pd
from src.load_data import load_csv, load_ventas_byproduct, load_promos_range, load_stock_byproduct, load_data


profiling = False
if profiling:
    from pandas_profiling import ProfileReport

# PRINT EDA FOR THE ORIGINAL CSV  
for name in ["ventas", "promos", "stock","prevision", "festivos"]:

    df = load_csv(name)
    
    if name == "promos":
        df =df[['id', 'iniciopromo', 'finpromo', 'semanainicio', #'semanafin',
        'producto', 'preciotarifa', 'preciopromocion', #'cantidad', 
        'tipopromo']]
     
    df.to_csv("data/clean/"+ name + ".csv", index=False)
    if profiling:
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

from pandas_profiling import ProfileReport
from src.load_data import load_data

def print_df(df):
    print(df.shape)
    print(df.head())
    print(df.info())

dfs = load_data()
for df, name in zip(dfs, ["ventas", "promos", "stock","prevision", "festivos"]):
    
    if name == "promos":
        df =df[['id', 'iniciopromo', 'finpromo', 'semanainicio', #'semanafin',
        'producto', 'preciotarifa', 'preciopromocion', #'cantidad', 
        'tipopromo']]
    print_df(df)
      
    profile = ProfileReport(df, title='Pandas Profiling Report')
    profile.to_file(output_file="reports/eda/"+name+".html")


ventas, promos, stock, prevision, festivos = dfs
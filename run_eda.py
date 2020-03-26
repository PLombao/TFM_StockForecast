from pandas_profiling import ProfileReport
from src.load_data import load_data

def print_df(df):
    print(df.shape)
    print(df.head())
    print(df.info())

dfs = load_data()
for df in dfs:
    print_df(df)
    profile = ProfileReport(df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
    profile.to_file(output_file="your_report.html")

ventas, promos, stock, prevision, festivos = dfs
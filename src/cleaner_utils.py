import pandas as pd 

def check_len_ts(ts, datefield):
    """
    Check the lenght of a time series
    """
    # Load festivos
    festivos = pd.read_csv("data/clean/festivos.csv", parse_dates=["fecha"], dayfirst=True)
    
    # Rango de fechas desde inicio hasta fin
    rng = pd.date_range(ts[datefield].min(), ts[datefield].max(), freq='D')

    # Fechas para las que hay datos
    dates = list(ts[datefield].drop_duplicates())

    # Construimos un dataframe con las fechas faltantes
    missing_dates = []
    for date in rng:
        if date not in dates:
            missing_dates.append(date)
    # Built df with missing, dates, weekdays and festivos
    miss_dates = pd.DataFrame({"fecha":missing_dates})
    miss_dates['weekday'] = miss_dates.fecha.apply(lambda dt: dt.weekday())
    miss_dates = miss_dates.merge(festivos, how='left').fillna(0)

    print("Number of dates missings:                                {}".format(miss_dates.shape[0]))
    miss_work_dates = miss_dates.loc[(miss_dates.festivo != 1) & (miss_dates.weekday != 6)]
    print("Number of dates missings (droping sundays and festivos): {}".format(miss_work_dates.shape[0]))

    return miss_work_dates.reset_index(drop=True)
    

def clean_csv(df, datecol):
    print('{:=^40}'.format('  CLEAN CSV  '.format()))
    # Correct columns for all lowcase
    df.columns = [col.lower() for col in df.columns]
    print("Set columns to lowcase.")

    # Sort by date
    df = df.sort_values(datecol)
    print("Sort values by date.")

    # Drop duplicates
    filter_data = df.drop_duplicates()
    print("Dropped duplicates. Rows dropped: {}."\
        .format(df.shape[0]-filter_data.shape[0]))

    print('{:=^40}'.format(''.format()))
    return filter_data
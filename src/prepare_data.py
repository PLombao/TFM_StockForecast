#### HELPERS FUNCTIONS TO PREPARE DATA FOR TRAINING MODELS
#### 1) Filter train data
#### 2) Split data into train and test

from sklearn.model_selection import train_test_split


def _drop_missings(data):
    """
    Function to drop the missings that can not be assigned
    Args:
        data (pd.DataFrame):    dataframe
    Returns:
        data (pd.DataFrame):    a filtered dataframe with only assignable missings
    """
    # Dropping missing of days without correct data
    filter_data = data.loc[data.fecha < '2020-03-23'].reset_index(drop=True)
    print("Dropped rows corresponding to 23 to 26-03-2020 for not having the ventas data for these days.")
    print("Rows dropped: {}".format(data.shape[0] - filter_data.shape[0]))

    # Dropping missing of type 2 (periods longer than one day)
    # filter_data2 = filter_data.loc[filter_data.stockMissingType != 2].reset_index(drop=True)
    # print("Dropped rows corresponding to missing from periods longer than one day.")
    # print("Rows dropped: {}".format(filter_data.shape[0] - filter_data2.shape[0]))

    return filter_data

def _assing_missings_venta(df, col):
    """Assign missings: 0 for holiday
                        4wd rolling windows for else
    """
    print("Assigning missings for {}".format(col))
    data = df.copy()
    holiday_data = data.loc[(data['festivo'] == 1) | (data['weekday'] == 6)]
    wd_data = data.loc[(data['festivo'] != 1) & (data['weekday'] != 6)]
    
    print("Missing in dataset:               {} ({} total rows).".format(sum(data[col].isna()), data.shape[0]))
    print("Missings in holiday days:         {} ({} total rows).".format(sum(holiday_data[col].isna()), holiday_data.shape[0]))
    print("Missings in working days:         {} ({} total rows).".format(sum(wd_data[col].isna()), wd_data.shape[0]))
    
    data.loc[(data['festivo'] == 1) | (data['weekday'] == 6), col] = holiday_data[col].fillna(0)
    print("Assigned missings for holiday data - Remaining missings:      {}".format(sum(data[col].isna())))
    
    data.loc[(data['festivo'] != 1) & (data['weekday'] != 6), col] = wd_data[col].fillna(wd_data["roll4wd_" + col])
    print("Assigned missings for working days data - Remaining missings: {}".format(sum(data[col].isna())))
    return data

def _assing_missings_stock(df, col):
    """Assign missings: backfill for holiday
                        4wd rolling windows for else???
    """
    print("Assigning missings for {}".format(col))
    data = df.copy()
    holiday_data = data.loc[(data['festivo'] == 1) | (data['weekday'] == 6)]
    wd_data = data.loc[(data['festivo'] != 1) & (data['weekday'] != 6)]
    
    print("Missing in dataset:               {} ({} total rows).".format(sum(data[col].isna()), data.shape[0]))
    print("Missings in holiday days:         {} ({} total rows).".format(sum(holiday_data[col].isna()), holiday_data.shape[0]))
    print("Missings in working days:         {} ({} total rows).".format(sum(wd_data[col].isna()), wd_data.shape[0]))
    
    data['holiday_' + col] = data.fillna(method='bfill')
    print("Assigned missings for holiday data - Remaining missings:      {}".format(sum(data[col].isna())))
    
    data.loc[(data['festivo'] != 1) & (data['weekday'] != 6), col] = wd_data[col].fillna(wd_data["roll4wd_" + col])
    print("Assigned missings for working days data - Remaining missings: {}".format(sum(data[col].isna())))
    return data

def _assing_missings(data):
    """
    Function to assign missings
    """
    # Assign stock missings for holidays
    data.loc[(data.festivos == )]
    data['udsstock'] = data['udsstock'].fillna(method='pad', limit=1)
    print("Assigned missings in udstock with fill forward method with 1 period limit")

    # Assing missings in uds venta
    data['udsventa'] = data['udsventa'].fillna(0)
    print("Assigned missings in udsventa.")

    # Assing missings in uds prevision as 0
    data['udsprevisionempresa'] = data['udsprevisionempresa'].fillna(0)
    print("Assigned missings in udsprevisionempresa filling with 0")


    return data

def prepare_train_data(data):
    """
    Filter the dataframe for the model given a producto
    Args:
        data (pd.DataFrame):    dataframe from clean/stock_all
    Returns:
        data (pd.DataFrame):    a filtered dataframe with only valid rows to train
    """
    print('{:=^60}'.format('  FILTER TRAIN DATA  '))
    data = _drop_missings(data)
    data = _assing_missings(data)
    print("Output shape: {}".format(data.shape))
    print('{:=^60}'.format(''))
    return data

def split_data(data, target, test_size=0.25):
    """
    Split data into: train_x, train_y, test_x, test_y. 
    Args:
        data (pd.DataFrame):    dataframe loaded from the table "prodiq_mft_value_lab"
        target (str):           two values are posible:  "lab_density_avg", "lab_ib_avg"
        test_size (float):      size (between 0-1) of the test set
    Returns:
        train_x (pd.DataFrame): a dataframe that contains the train data with all posible predictor variables.
        train_y (np.array): an array that contains the train data with target variable.
        test_x (pd.DataFrame): a dataframe that contains the test data with predictor variables.
        test_y (np.array): an array that contains the test data with target variable.
    """
    print("- Splitting the data with test proportion: {}% ...".format(test_size*100))

    # Split the data into training and test sets
    train, test = train_test_split(data, test_size=test_size)
    # Split the data into training and tests with test equal to last 5 registers 
    #train, test = train_test_split(data, test_size=test_size, shuffle = False)

    # Split the data into predictor and target variables
    train_x = train.drop([target], axis=1)
    test_x = test.drop([target], axis=1)
    train_y = train[[target]]
    test_y = test[[target]]
    print("Train size: {}".format(train.shape))
    print("Test size: {}".format(test.shape))
    return train_x, train_y, test_x, test_y
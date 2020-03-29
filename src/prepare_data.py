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
    return data

def filter_train_data(data, producto):
    """
    Filter the dataframe for the model given a producto
    Args:
        data (pd.DataFrame):    dataframe from clean/stock_all
        producto (str):         the name of the product
    Returns:
        data (pd.DataFrame):    a filtered dataframe with only valid rows to train
    """
    data = _drop_missings(data)
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
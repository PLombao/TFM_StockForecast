from src.load_data import load_data
from src.prepare_data import filter_train_data
from src.trainer import run, run_cv
if __name__ == "__main__":

    # Load Stock data
    data = load_data()

    # Prepare data
    # For now there is no prepare data for producto
    data = filter_train_data(data)

    # Initalize base model
    from sklearn.linear_model import LinearRegression
    base_model = LinearRegression()

    if False:
        pass
    else:
        print("DEMO MODE")
        print("")
        data = data.loc[data.producto == "1", ["udsventa", "udsstock", "weekday"]]

        data = data.fillna(0).reset_index(drop=True)

        metrics = run_cv(data, "udsstock", base_model, "Model1")




        
        print(data.head())
        print(data.shape)
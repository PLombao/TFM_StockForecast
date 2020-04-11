from src.load_data import load_data
from src.prepare_data import prepare_train_data
from src.trainer import run, run_cv
if __name__ == "__main__":

    # Load Stock data
    data = load_data()

    # Initalize base model
    from sklearn.linear_model import LinearRegression
    base_model = LinearRegression()

    if False:
        pass
    else:
        print("DEMO MODE")
        print("")
        data = data.loc[data.producto == "1"]

        # Prepare data
        data = prepare_train_data(data)
        data = data[["udsventa", "udsstock", "weekday"]]

        metrics = run_cv(data, "udsstock", base_model, "Model1")




        
        print(data.head())
        print(data.shape)
import json
import sys

def read_source_data(dataset):
    json_path = "./config/source_data.json"
    with open("./config/source_data.json") as config_file: 
        src_data = json.load(config_file)
    
    # Check if dataset configured
    if dataset not in list(src_data): 
        print("[ERROR] Dataset {} not configured in {}.".format(dataset, json_path))
        sys.exit()
    else:
        dataset_data = src_data[dataset]

        # Get filename
        filename = dataset_data["filename"]

        # Get columns type
        dates = dataset_data["dates"]
        columns = dataset_data["columns"]

        # Get dataframe shape
        rows = dataset_data["rows"]
        cols = len(dates) + len(list(columns))
        shape = (rows, cols)

        return [filename, dates, columns, shape]


def read_config_model(modelo, name):
    """
    Read config.json file
    
    Args:
        modelo (str):   two possible values: "stock" or "clustering"
        name (str):     model name in the json
    Returns:
        list_var_model (list):          a list with the variables which use the model
        dict_params_model (dictionary): a dictionary with the params of the model (base_learner)
        dict_tags_model (dictionary):   a dictionary with the tags of the model (to mlflow)
    """
    if modelo in ["stock", "clustering"]:
        json_path = "./config/model_" + modelo + ".json"

        with open(json_path) as config_file: 
            config = json.load(config_file)
            info_model_dict = config[name]
            variables = info_model_dict['variables']
            tags = info_model_dict['tags']
            params = info_model_dict['params']
        
        return variables, tags, params
    else:
        print("[ERROR] File ./config/model_" + modelo + ".json doesn't exist")
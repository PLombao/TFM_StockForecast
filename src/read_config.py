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

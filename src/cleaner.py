import numpy as np
import pandas as pd

from src.create_variables import create_variables

def cleaner(data):
    print('{:=^60}'.format('  CLEANER DATASET STOCK  '))
    print("Input shape: {}".format(data.shape))
    data = create_variables(data)
    print('{:=^60}'.format(''))
    return data
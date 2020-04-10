import numpy as np
import pandas as pd

from src.create_variables import create_variables

def cleaner(data):
    data = create_variables(data)

    return data
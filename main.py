import sys, os

import sklearn as sl
import pandas as pd
import numpy as np
import matplotlib as plot
import seaborn as sb

import data_cleaning as cd
import data_processing as dp
import data_visualization as dv

def get_data(filename):
    """
    Read the given CSV file. Returns DataFrame.
    """
    df = pd.read_csv(filename, parse_dates=['date_confirmation'])
    return df


def main(train_data, test_data, out_directory):
    train = get_data(train_data)
    test = get_data(test_data)

    


if __name__ == '__main__':
    
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    out_directory = sys.argv[3]
    main(train_data, test_data, out_directory)
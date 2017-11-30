import warnings
import pandas as pd

warnings.filterwarnings('ignore')

test_path="../data/test.csv"
train_path="../data/train.csv"

def getDataCSV(csvPath):
    data=pd.read_csv(csvPath)
    return data

def downsize(var,scale):
    for i in range(0,len(var)):
        var[i] = var[i]/scale

    return var
import pandas as pd 
import plotly.express as px 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np  
from sklearn.metrics import root_mean_squared_error
import optuna
import xgboost as xgb
from sklearn.linear_model import LinearRegression 

from pathlib import Path  # path library 

def rmse(predicted, target): 
    return np.sqrt(np.mean((predicted - target) ** 2))

data_dir = Path(__file__).resolve().parent.parent 

train_df = pd.read_parquet(data_dir / 'train.parquet') 
val_df = pd.read_parquet(data_dir  / 'val.parquet')  # được tách băng train_test_split từ train 
test_df = pd.read_parquet(data_dir / 'test.parquet')

class MeanRegressor(): 
    def fit(self, inputs, targets): 
        self.__mean = np.mean(targets)
    def predict(self, inputs):
        return np.full(inputs.shape[0], self.__mean)  # inputs.shape sẽ trả về np array (tuple với data frame) chứa độ lớn của các chiều

target_col = "fare_amount"
drop_cols = ["key", "fare_amount", "pickup_datetime"]
input_cols = [c for c in train_df.columns if c not in drop_cols]

model = MeanRegressor()
model.fit(train_df[input_cols], train_df[target_col]) 
hard_predict = model.predict(val_df[input_cols]) 
print(rmse(hard_predict, val_df[target_col])) 
# 9.662449 (hard_model)


linear_model = LinearRegression().fit(train_df[input_cols], train_df[target_col]) 
print(rmse(linear_model.predict(val_df[input_cols]), val_df[target_col]))
# 4.970677918117792 (Linear Regression)


import pandas as pd 
import plotly.express as px 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np  
from sklearn.metrics import root_mean_squared_error
import optuna
import xgboost as xgb

from pathlib import Path  # path library 

data_dir = Path(__file__).resolve().parent.parent 

train_df = pd.read_parquet(data_dir / '/train.parquet') 
val_df = pd.read_parquet(data_dir  / '/val.parquet')  
test_df = pd.read_parquet(data_dir / '/test.parquet')

# print(train_df.columns)
target_col = 'fare_amount' 
drop_cols = ['key', 'fare_amount', 'pickup_datetime']
input_cols = [c for c in train_df.columns if c not in drop_cols] 


train_inputs = train_df[input_cols]
train_targets = train_df[target_col]

val_inputs = val_df[input_cols]
val_targets = val_df[target_col]

test_inputs = test_df[input_cols] 

 
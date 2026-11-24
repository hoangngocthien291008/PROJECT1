import pandas as pd 
import plotly.express as px 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np  
from sklearn.metrics import root_mean_squared_error
import optuna
import xgboost


from pathlib import Path 

data_dir = Path(__file__).resolve().parent.parent 

import pyarrow 

test_df = pd.read_parquet(data_dir / "test.parquet") 
train_df = pd.read_parquet(data_dir / "train.parquet")
val_df = pd.read_parquet(data_dir / "val.parquet")

target_col = "fare_amount" 
drop_cols = ["key", "fare_amount", "pickup_datetime"]
input_cols = [c for c in train_df.columns if c not in drop_cols] 

test_inputs = test_df[input_cols] 

train_inputs = train_df[input_cols] 
train_targets = train_df[target_col] 

val_inputs = val_df[input_cols] 
val_targets = val_df[target_col] 


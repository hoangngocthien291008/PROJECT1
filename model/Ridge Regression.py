import pandas as pd 
import plotly.express as px 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np  
from sklearn.metrics import root_mean_squared_error
import optuna
import xgboost

from pathlib import Path  # path library 

data_dir = Path(__file__).resolve().parent.parent 

def rmse(predict, target): 
    return np.sqrt(np.mean((predict - target) ** 2)) 

def evaluate(model): 
    train_preds = model.predict(train_inputs)
    val_preds = model.predict(val_inputs)
    return {"Train_error" : rmse(train_preds, train_targets), "Val_error" : rmse(val_preds, val_targets)}

def predict_and_submit(model, test_set, fname): 
    test_preds = model.predict(test_set)
    submission_df = pd.read_csv(data_dir / "new-york-city-taxi-fare-prediction/sample_submission.csv")
    submission_df["fare_amount"] = test_preds 
    submission_df.to_csv(data_dir / f"submission/{fname}.csv", index = False)


train_df = pd.read_parquet(data_dir / 'train.parquet') 
val_df = pd.read_parquet(data_dir  / 'val.parquet')  
test_df = pd.read_parquet(data_dir / 'test.parquet')

# print(train_df.columns)
target_col = 'fare_amount' 
drop_cols = ['key', 'fare_amount', 'pickup_datetime']
input_cols = [c for c in train_df.columns if c not in drop_cols] 


train_inputs = train_df[input_cols]
train_targets = train_df[target_col]

val_inputs = val_df[input_cols]
val_targets = val_df[target_col]

test_inputs = test_df[input_cols] 

from sklearn.linear_model import Ridge
model1 = Ridge(random_state = 42) 
model1.fit(train_inputs, train_targets) 

set_error = evaluate(model1) 

for inputs, error in set_error.items(): 
    print(f"{inputs} : {error:.4f}", end = "\n") 

"""
Train_error : 4.9187
Val_error : 4.9694
"""

predict_and_submit(model1, test_inputs, "Ridge") 

# score 4.47798 in kaggle


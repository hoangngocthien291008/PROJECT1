import pandas as pd 
import plotly.express as px 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np  
from sklearn.metrics import root_mean_squared_error
import optuna
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

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
 

def objective(trial): 
    # search param 
    params = {
        "max_features": trial.suggest_categorical("max_features", ['sqrt', 0.5, 0.7, 1.0]),
        "max_depth": trial.suggest_int("max_depth", 12, 25), 
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 30),
        "n_estimators": 50,
        "max_samples": 0.4, # dữ liệu đầu vào 
        "n_jobs": -1,
        "random_state": 42
    }
    # our model 
    model = RandomForestRegressor(**params)
    # do cross validation 
    model.fit(train_inputs, train_targets)
    val_preds = model.predict(val_inputs)
    # return score 
    return rmse(val_preds, val_targets)

study = optuna.create_study(direction = "minimize") 
study.optimize(objective, n_trials = 25, show_progress_bar = True, gc_after_trial = True) 

final_params = study.best_params.copy()
final_params["n_estimators"] = 200  # Tăng số cây cho model cuối
final_params["max_samples"] = 0.8   # Tăng dữ liệu học mỗi cây lên 80%

model2 = RandomForestRegressor(n_jobs = -1, random_state = 42, verbose = False, **final_params) # chỉ đưa ra các tham số đc suggest
model2.fit(train_inputs, train_targets)

evaluate_model = evaluate(model2)
for name, error in evaluate_model.items(): 
    print(f"The error of {name}: {error:.4f}") 

predict_and_submit(model2, test_inputs, "RandomForest")

"""
The error of Train_error: 2.8427
The error of Val_error: 3.7016
"""

# score 3.19398 in kaggle
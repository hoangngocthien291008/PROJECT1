import pandas as pd 
import numpy as np  
import optuna
from sklearn.ensemble import RandomForestRegressor
import utils

train_inputs, train_targets, val_inputs, val_targets, test_inputs = utils.load_data()
 

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
    return utils.rmse(val_preds, val_targets)

study = optuna.create_study(direction = "minimize") 
study.optimize(objective, n_trials = 25, show_progress_bar = True, gc_after_trial = True) 

final_params = study.best_params.copy()
final_params["n_estimators"] = 200  # Tăng số cây cho model cuối
final_params["max_samples"] = 0.8   # Tăng dữ liệu học mỗi cây lên 80%

model2 = RandomForestRegressor(n_jobs = -1, random_state = 42, verbose = False, **final_params) # chỉ đưa ra các tham số đc suggest
model2.fit(train_inputs, train_targets)

evaluate_model = utils.evaluate(model2)
for name, error in evaluate_model.items(): 
    print(f"The error of {name}: {error:.4f}") 

utils.predict_and_submit(model2, test_inputs, "RandomForest")

"""
The error of Train_error: 2.8427
The error of Val_error: 3.7016
"""

# score 3.19398 in kaggle

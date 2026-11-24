import pandas as pd 
import optuna
from xgboost import XGBRegressor
import numpy as np  
import utils 

train_inputs, train_targets, val_inputs, val_targets, test_inputs = utils.load_data() 

def objective(trial): 
    params = {
        "n_estimators" : trial.suggest_int("n_estimators", 300, 1500), 
        "max_depth" : trial.suggest_int("max_depth", 5, 12), 
        "learning_rate" : trial.suggest_float("learning_rate", 0.01, 0.3, log = True), # log = True để phân chia đề theo số mũ log
        "gamma" : trial.suggest_categorical("gamma", [0, 1e-7, 1e-8, 1e-5, 1e-6]), 
        "subsample" : trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 0.9), 
        "reg_alpha" : trial.suggest_float("reg_alpha", 0.1, 10, log = True), 
        "reg_lambda" : trial.suggest_float("reg_lambda", 0.1, 10, log = True), # phân bố đều hơn theo số mũ 
        "early_stopping_rounds" : 50,
        "n_jobs" : -1, 
        "device" : "cuda", 
        "random_state" : 42, 
        "verbosity" : 0
    }

    model = XGBRegressor(**params) 
    model.fit(train_inputs, train_targets, eval_set = [(val_inputs, val_targets)], verbose = False)
    trial.set_user_attr("best_iteration", model.best_iteration) # số vòng lặp cần trước khi bị dừng lại do early_stopping_rounds
    val_preds = model.predict(val_inputs) 
    return utils.rmse(val_preds, val_targets)

study = optuna.create_study(direction = "minimize")
study.optimize(objective, n_trials = 30, show_progress_bar = True, gc_after_trial = True) 

needed_iteration = study.best_trial.user_attrs["best_iteration"]
final_params = study.best_params.copy()
final_params["n_estimators"] = needed_iteration 


# chỉ cần làm tới khi bị early stopping rounds dừng lại thôi --> tối ưu tài nguyên và tránh overfit 
model3 = XGBRegressor(**final_params, verbosity = 0, random_state = 42, device = "cuda", n_jobs = -1) 
model3.fit(train_inputs, train_targets)
evaluate_model3 = utils.evaluate(model3, train_inputs, train_targets, val_inputs, val_targets)

for name, error in evaluate_model3.items(): 
    print(f"The error of {name}: {error:.4f}")

utils.predict_and_submit(model3, test_inputs, "XGBoost")

"""
The error of Train_error: 2.4559
The error of Val_error: 3.6703
"""
{'n_estimators': 1310, 'max_depth': 10, 'learning_rate': 0.013922099514503785, 'gamma': 1e-05, 
 'subsample': 0.7594491193684779, 'colsample_bytree': 0.5654504078836895, 'reg_alpha': 5.781955239113914, 'reg_lambda': 0.866410641571653}
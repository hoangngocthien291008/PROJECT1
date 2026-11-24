import pandas as pd 
import plotly.express as px 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.metrics import root_mean_squared_error
import optuna
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd 
import numpy as np  
import utils 

train_inputs, train_targets, val_inputs, val_targets, test_inputs = utils.load_data() 

def objective(trial): 
    params = {

    }

    model = XGBRegressor() 
    model.fit(train_inputs, train_targets)
    val_preds = model.predict(val_inputs) 
    return utils.rmse(val_preds, val_targets)

study = optuna.create_study(direction = "minimize")
study.optimize(objective, n_trials = 30, show_progress_bar = True, gc_after_trial = True) 


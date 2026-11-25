import pandas as pd 
import numpy as np  
import optuna
import utils

train_inputs, train_targets, val_inputs, val_targets, test_inputs = utils.load_data() 

from sklearn.linear_model import Ridge
model1 = Ridge(random_state = 42) 
model1.fit(train_inputs, train_targets) 

set_error = utils.evaluate(model1) 

for inputs, error in set_error.items(): 
    print(f"{inputs} : {error:.4f}", end = "\n") 

"""
Train_error : 4.9187
Val_error : 4.9694
"""

utils.predict_and_submit(model1, test_inputs, "Ridge") 

# score 4.47798 in kaggle


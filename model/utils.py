import pandas as pd 
import numpy as np  
import pyarrow 
from pathlib import Path 

data_dir = Path(__file__).resolve().parent.parent 


def rmse(predict, target): 
    return np.sqrt(np.mean((predict - target) ** 2)) 

def evaluate(model, train_inputs, train_targets, val_inputs, val_targets): 
    train_preds = model.predict(train_inputs)
    val_preds = model.predict(val_inputs)
    return {"Train_error" : rmse(train_preds, train_targets), "Val_error" : rmse(val_preds, val_targets)}

def predict_and_submit(model, test_set, fname): 
    test_preds = model.predict(test_set)
    submission_df = pd.read_csv(data_dir / "new-york-city-taxi-fare-prediction/sample_submission.csv")
    submission_df["fare_amount"] = test_preds 
    submission_df.to_csv(data_dir / f"submission/{fname}.csv", index = False)

def load_data(): 
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
    return train_inputs, train_targets, val_inputs, val_targets, test_inputs


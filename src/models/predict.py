import os
import mlflow
import pandas as pd

from src.utils.config import final_features

PATH = os.path.abspath("src/models/model/best_model/artifacts")

try:
    model = mlflow.pyfunc.load_model(PATH)
    print("model loaded successfully")
except Exception as e:
    print("Failed to load model")

def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=final_features,fill_value=0)

    bool_cols = [c for c in df.columns if c not in ["age", "balance"]]

    df["age"] = df["age"].astype(float)
    df["balance"] = df["balance"].astype(float)

    df[bool_cols] = df[bool_cols].astype(bool)

    try:
        preds = model.predict(df)
        
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  
            
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    if result == 1:
        return "Likely to subscribe" 
    else:
        return "Not likely to subscribe" 
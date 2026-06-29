import pandas as pd
from src.utils import config
from sklearn.preprocessing import RobustScaler


class DataPreprocessor:
    def __init__(self, df):
        self.scaler = RobustScaler()

        self.scaler.fit(df[config.numeric_features])
        print("Fitted scaler")


    def preprocess(self, df: pd.DataFrame, target_col: str = "y") -> pd.DataFrame: 
        name = df.__class__.__name__
        """
        Basic cleaning for Telemarketing data
        - Map Target to 0/1
        - Keep relevant columns
        - Mapping of job groups
        - Perform scaling using RobustScaler
        """

        df[target_col] = df[target_col] == "yes"
        print(f"Mapped target column for {name}")

        df["job_group"] = (
            df["job"]
            .map(config.job_mapping)
            .fillna("unknown")
        )
        print(f"Mapped job column for {name}")

        df = df[config.relevant_columns]
        print(f"Kept relevant columns for {name}")

        df[config.numeric_features] = self.scaler.transform(
            df[config.numeric_features]
        )
        print(f"Scaled numeric features for {name}")

        return df[config.relevant_features], df[target_col]

        
    

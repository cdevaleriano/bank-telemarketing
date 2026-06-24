import pandas as pd
import os

def ingest_csv(path: str) -> pd.DataFrame:
    """
    Loads CSV data into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.    
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    return pd.read_csv(path)
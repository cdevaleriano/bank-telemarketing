from pathlib import Path
import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(Path(path))
        print(f"Loaded dataset: {path}")
        return df

    except FileNotFoundError:
        print(f"Dataset not found.")
        raise

    except Exception:
        print("Unexpected error while loading dataset.")
        raise
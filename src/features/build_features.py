"""
Build features
"""

import pandas as pd

def build_features(df):
    df = pd.get_dummies(df, drop_first=True)
    print(f"built features: one-hot encoding")

    if 'contact_unknown' not in df.columns:
        df['contact_unknown'] = False

    return df
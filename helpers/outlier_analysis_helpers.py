import pandas as pd
from scipy.stats import zscore

def detect_outliers_zscore(df, column, threshold=3) -> pd.DataFrame:
    """
    Calculate the zscore for each column, and return rows where zscore
    exceeds the threshold, assumed to be outliers
    """
    df['zscore'] = zscore(df[column])
    return df[abs(df['zscore']) > threshold]
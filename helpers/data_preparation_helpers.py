from sklearn.preprocessing import StandardScaler
import pandas as pd

def standardise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to standardise the numeric columns of a dataframe.
    This ensures a mean of 0 and standard deviation of 1.
    It mains the data distribution of the dataset, and prepares for model training.
    """
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['float']).columns
    df.copy()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
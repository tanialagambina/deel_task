import pandas as pd

def whitespace_remover(dataframe) -> pd.DataFrame:
    """
    Function to clean string data in dataframe by removing whitespace
    """
    for i in dataframe.columns:
        if dataframe[i].dtype == 'object':
            dataframe[i] = dataframe[i].map(str.strip)
        else:
            pass
    return dataframe
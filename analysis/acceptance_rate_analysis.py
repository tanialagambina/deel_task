import pandas as pd

def get_acceptance_rate(df: pd.DataFrame, to_group: list) -> pd.DataFrame:
    """
    Function to apply acceptance rate calculation to a dataframe and return results
    """
    return df.groupby(to_group).apply(calculate_acceptance_rate).reset_index().rename(
        columns={0: 'acceptance_ratio'})

def calculate_acceptance_rate(group) -> float:
    """
    Function to calculate the acceptance rate on a group.
    Acceptance rate is defined as the number of accepted transactions divited by
    the total attempted transactions.
    """
    accepted_count = group['state'].value_counts().get('ACCEPTED', 0)
    total_count = len(group['state'])
    return accepted_count / total_count

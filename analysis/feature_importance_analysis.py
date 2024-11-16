import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def calculate_chi2(data, outcome_var) -> pd.DataFrame:
    """
    Function to apply chi-squared analysis to a dataframe and return
    the results sorted in order of significance
    """
    results = {}
    for col in data.columns:
        if col != outcome_var:
            contingency_table = pd.crosstab(data[col], data[outcome_var])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results[col] = p
    return pd.DataFrame({'Variable': results.keys(), 'p_value': results.values()}).sort_values('p_value')


def calculate_cramers_v(df) -> pd.DataFrame:
    """
    Function to apply cramers v analysis to a dataframe and return
    the results sorted in order of significance
    """
    results = []
    for col in df.columns:
        contingency_table = pd.crosstab(df[col], df['state'])
        cramers_v_value = cramers_v(contingency_table.values)
        results.append({'column': col, 'cramers_v': cramers_v_value})
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='cramers_v', ascending=False).reset_index(drop=True)
    return results_df


def cramers_v(confusion_matrix) -> float:
    """
    Function to perform the cramers_v calculation
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    denominator = n * (min(r, k) - 1)
    if denominator == 0 or denominator < 0:
        return np.nan  # Return NaN if division is invalid
    return np.sqrt(chi2 / denominator)
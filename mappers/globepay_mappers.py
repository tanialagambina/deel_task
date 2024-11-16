from pydantic import ValidationError
from models.acceptance_report import AcceptanceReport
from models.chargeback_report import ChargebackReport
import pandas as pd
import json

def map_acceptance_report(acceptance_report: pd.DataFrame) -> pd.DataFrame:
    """
    Function to map the acceptance report data to the correct data types and perform
    validation checks
    """
    validated_data = acceptance_report.apply(validate_acceptance_report, axis=1)
    validated_data = validated_data.dropna()
    validated_df = pd.DataFrame(validated_data.tolist())
    return validated_df


def map_chargeback_report(chargeback_report: pd.DataFrame) -> pd.DataFrame:
    """
    Function to map the chargeback report data to the correct data types and perform
    validation checks
    """
    validated_data = chargeback_report.apply(validate_chargeback_report, axis=1)
    validated_data = validated_data.dropna()
    validated_df = pd.DataFrame(validated_data.tolist())
    return validated_df


def validate_acceptance_report(row):
    """
    Function to run through each row of the acceptance report, and to
    validate it follows the AcceptanceReport data type
    """
    row_data = row.to_dict()
    row_data['external_ref'] = str(row_data['external_ref'])
    row_data['source'] = str(row_data['source'])
    row_data['ref'] = str(row_data['ref'])
    row_data['date_time'] = pd.to_datetime(row_data['date_time'], format="%Y-%m-%dT%H:%M:%S.%fZ")
    row_data['state'] = str(row_data['state'])
    row_data['country'] = str(row_data['country'])
    row_data['currency'] = str(row_data['currency'])
    row_data['rates'] = json.loads(row_data['rates'])
    row_data['amount'] = float(row_data['amount'])
    try:
        report = AcceptanceReport(**row_data)
        return report.dict()
    except ValidationError as e:
        print(f"Validation error for row: {row_data}")
        print(e)
        return None


def validate_chargeback_report(row):
    """
    Function to run through each row of the chargeback report, and to
    validate it follows the ChargebackReport data type
    """
    row_data = row.to_dict()
    try:
        report = ChargebackReport(**row_data)
        return report.dict()
    except ValidationError as e:
        print(f"Validation error for row: {row_data}")
        print(e)
        return None
from typing import Dict, List, Optional
import pandas as pd
from logging import getLogger

logger = getLogger(__name__)
log = logger

DATA_TYPE_MAPPING = {
    "text": ["MCC", "Vendor", "AgencyName"],
    "categorical": ["AgencyNumber"],
    "numerical": ["Amount", "Timestamp"],
    "index": "AgencyName",
}

DEFAULT_RENAME_MAPPING = {
    'Agency Number': 'AgencyNumber',
    'Transaction Date': 'Timestamp',
    'Merchant Category Code (MCC)': 'MCC',
    'Vendor': 'Vendor',
    'Amount': 'Amount',
    'Agency Name': 'AgencyName',
}

INVERSE_RENAME_MAPPING = {v: k for k, v in DEFAULT_RENAME_MAPPING.items()}

DEFAULT_COLUMN_ORDER = ['AgencyName', 'Vendor', 'MCC', 'Timestamp', 'Amount']


def print_dataset_summary(info_dict: Dict[str, int]):
    print("-" * 50)
    print("Dataset Summary:")
    for key, value in info_dict.items():
        print(f"{key}: {value}")
    print("-" * 50)

class NuTable:    
    @staticmethod
    def clean_unnamed(
            df: pd.DataFrame
        ):
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        before_len = len(df)
        for col in unnamed_cols:
            df = df[pd.isna(df[col])]
        after_len = len(df)
        log.info(f"Removed {before_len - after_len} rows with unnamed columns")
        df.drop(columns=unnamed_cols, inplace=True)
        return df

    @staticmethod    
    def clean_discrepant_names(
            df: pd.DataFrame,
        ):
        discrepant_names_df = pd.DataFrame(df.groupby(['Agency Number'])['Agency Name'].unique()).reset_index()
        discrepant_names = discrepant_names_df[discrepant_names_df['Agency Name'].apply(lambda x: len(x) > 1)].copy()
        # this may not be the best way to select the names, but 
        # saves us from having to manually check the names and hard code the desired outputs
        discrepant_names['Correct Name'] = [name[0] for name in discrepant_names['Agency Name']]
        for agency_number in discrepant_names['Agency Number']:
            correct_name = discrepant_names.loc[discrepant_names['Agency Number'] == agency_number, 'Correct Name'].values[0]
            df.loc[df['Agency Number'] == agency_number, 'Agency Name'] = correct_name
        return df

    @staticmethod
    def clean_dates(
            df: pd.DataFrame,
        ):
        # clean dates that are not in the format YYYY/MM/DD
        df.dropna(subset=['Transaction Date'], inplace=True)
        df['Transaction Date'] = df['Transaction Date'].apply(lambda x: x if '/' in x else None)
        df.dropna(subset=['Transaction Date'], inplace=True)
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
        return df

    @staticmethod
    def rename_columns(
            df: pd.DataFrame,
            column_map: Optional[Dict[str, str]] = None,
        ):
        if column_map is None:
            column_map = DEFAULT_RENAME_MAPPING
        df.rename(columns=column_map, inplace=True)
        return df
    
    @staticmethod
    def reorder_columns(
            df: pd.DataFrame,
            column_order: Optional[List[str]] = None
        ):
        if column_order is None:
            column_order = DEFAULT_COLUMN_ORDER
        df = df[column_order]
        return df

    @staticmethod
    def clean_all_and_rename(
            df: pd.DataFrame,
            column_map: Optional[Dict[str, str]] = None,
            column_order: Optional[List[str]] = None
        ):
        df_copy = df.copy()
        df_copy = NuTable.clean_unnamed(df_copy)
        df_copy = NuTable.clean_discrepant_names(df_copy)
        df_copy = NuTable.clean_dates(df_copy)
        df_copy = NuTable.rename_columns(df_copy, column_map)
        df_copy = NuTable.reorder_columns(df_copy, column_order)
        return df_copy
    
    @staticmethod
    def get_categorical_columns(
            data_type_mapping: Optional[Dict[str, List[str]]] = None,
        ):
        if data_type_mapping is None:
            data_type_mapping = DATA_TYPE_MAPPING
        return data_type_mapping["categorical"]
    
    @staticmethod
    def get_numerical_columns(
            data_type_mapping: Optional[Dict[str, List[str]]] = None,
        ):
        if data_type_mapping is None:
            data_type_mapping = DATA_TYPE_MAPPING
        return data_type_mapping["numerical"]
    
    @staticmethod
    def get_text_columns(
            data_type_mapping: Optional[Dict[str, List[str]]] = None,
        ):
        if data_type_mapping is None:
            data_type_mapping = DATA_TYPE_MAPPING
        return data_type_mapping["text"]
    
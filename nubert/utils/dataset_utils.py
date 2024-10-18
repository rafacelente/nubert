from typing import Dict, List, Optional
import math
import numpy as np
import pandas as pd
from logging import getLogger

logger = getLogger(__name__)
log = logger

DATA_TYPE_MAPPING = {
    "text": ["Merchant Category Code (MCC)", "Vendor", "Agency Name", "Amount"],
    "categorical": ["Agency Number"],
    "numerical": ["Timestamp"],
    "index": "Agency Name",
}

DEFAULT_COLUMN_ORDER = [
    'Agency Name',
    'Vendor',
    'Merchant Category Code (MCC)',
    'Timestamp',
    'Amount',
    'Transaction Date',
    'Original Amount',
    'Amount Min',
    'Amount Max',
]


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
    def filter_to_list_of_agency_names(
            df: pd.DataFrame,
            agency_list: List[str],
        ):
        column_name = 'Agency Name' if 'Agency Name' in df.columns else 'AgencyName'
        return df[df[column_name].isin(agency_list)]

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
            column_map: Dict[str, str],
        ):
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
    def clean_refunds(
        df: pd.DataFrame,
    ):
        return df[~(df['Amount'] < 0)]
    
    @staticmethod
    def strip_agencies(
        df: pd.DataFrame,
        agency_names: List[str] = ["EMPLOYEE BENEFITS"]
    ):
        for agency_name in agency_names:
            df = df[~df['Agency Name'].str.contains(agency_name)]
        return df
    
    @staticmethod
    def clean_amount(df: pd.DataFrame):
        df['Amount'] = df['Amount'].apply(lambda x: str(x).strip("$()").replace(",", "")).astype(float)
        return df
    
    @staticmethod
    def drop_columns(
        df: pd.DataFrame,
        columns: List[str] = ["Cardholder Last Name", "Cardholder First Initial", "Agency Number"],
    ):
        columns = [column for column in columns if column in df.columns]
        return df.drop(columns=columns)

    @staticmethod
    def quantize(data: np.ndarray, num_bins: int, min_value: float = 0.01, max_value: float = 1e7):
        log_min = math.log1p(min_value)
        log_max = math.log1p(max_value)
        bin_edges = np.logspace(log_min, log_max, num=num_bins+1, base=math.e) - 1
        bin_edges = np.concatenate([[-np.inf], [0], bin_edges, [np.inf]])

        def quantize(x):
            if x < 0:
                return 0  # Bin for negative values (refunds)
            elif x == 0:
                return 1  # Bin for zero values
            else:
                return np.digitize(x, bin_edges) - 1
        quant_inputs = np.array([quantize(x) for x in data])
        return quant_inputs, bin_edges
    
    @staticmethod
    def encode_timestamp(
        df: pd.DataFrame,
        time_column: str = "Transaction Date",
    ):
        df['Timestamp'] = pd.to_datetime(df[time_column]) - pd.to_timedelta(7, unit='d')
        df['Timestamp'] = df.groupby(pd.Grouper(key='Timestamp', freq='W-MON')).ngroup()
        return df

    @staticmethod
    def encode_amount(
        df: pd.DataFrame,
        num_bins: int = 20,
    ):
        df['Original Amount'] = df['Amount']
        df['Amount'], bin_edges = NuTable.quantize(df['Amount'], num_bins=num_bins)
        
        def _decode_quantized_amount(quantized_value: int, bin_edges: np.ndarray) -> tuple[float, float]:
            if quantized_value == 0:
                return -1.0, -1.0
            elif quantized_value == 1:
                return 0.0, 0.0
            else:
                lower_bound = bin_edges[quantized_value]
                upper_bound = bin_edges[quantized_value + 1]
                return lower_bound, upper_bound
        
        df['Amount Max'] = df['Amount'].apply(lambda x: _decode_quantized_amount(x, bin_edges)[1])
        df['Amount Min'] = df['Amount'].apply(lambda x: _decode_quantized_amount(x, bin_edges)[0])
        return df
    
    @staticmethod
    def fill_na(
        df: pd.DataFrame,
        fill_value: str = "Unknown",
    ):
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].fillna(fill_value)
            else:
                df[column] = df[column].fillna(0)
        return df

    # TODO: this should probably be called by a config
    @staticmethod
    def clean_all_and_rename(
            df: pd.DataFrame,
            column_map: Optional[Dict[str, str]] = None,
            column_order: Optional[List[str]] = None,
            agency_names_to_remove: List[str] = ["EMPLOYEE BENEFITS"],
            columns_to_drop: List[str] = ["Posted Date", "Year-Month", "Cardholder Last Name", "Cardholder First Initial", "Agency Number"],
            num_bins: int = 20,
        ):
        df_copy = df.copy()
        df_copy = NuTable.clean_unnamed(df_copy)
        df_copy = NuTable.clean_discrepant_names(df_copy)
        df_copy = NuTable.clean_dates(df_copy)
        df_copy = NuTable.clean_amount(df_copy)
        df_copy = NuTable.clean_refunds(df_copy)
        df_copy = NuTable.strip_agencies(df_copy, agency_names_to_remove)
        df_copy = NuTable.drop_columns(df_copy, columns_to_drop)
        df_copy = NuTable.fill_na(df_copy)
        df_copy = NuTable.encode_amount(df_copy, num_bins)
        df_copy = NuTable.encode_timestamp(df_copy)
        # df_copy = NuTable.rename_columns(df_copy, column_map)
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
    
    @staticmethod
    def validate_cleaned_data(df: pd.DataFrame) -> List[str]:
        warnings = []
        expected_columns = set(DEFAULT_COLUMN_ORDER)
        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            warnings.append(f"Missing columns: {', '.join(missing_columns)}")
        
        if 'Amount' in df.columns and (df['Amount'] < 0).any():
            warnings.append("Negative values found in 'Amount' column")
        
        if 'Transaction Date' in df.columns:
            try:
                pd.to_datetime(df['Transaction Date'])
            except ValueError:
                warnings.append("'Transaction Date' column contains invalid date formats")
        
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            warnings.append(f"NaN values found in columns: {', '.join(nan_columns)}")
        
        return warnings
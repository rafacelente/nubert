from typing import Literal, Optional
import os
from os import path
import pandas as pd
import numpy as np
import math
import tqdm
import pickle
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data.dataset import Dataset

from nugpt.utils import divide_chunks
from ..tokenizer import NuTokenizer
from ..utils import NuTable, DATA_TYPE_MAPPING

logger = logging.getLogger(__name__)
log = logger

class NuDataset(Dataset):
    def __init__(self,
                 model_name: str = "TinyLlama/TinyLlama_v1.1",
                 root: str = "./data/",
                 fname: str = "nudataset",
                 vocab_dir: str = "./data/vocab",
                 fextension: str = "",
                 user_ids: Optional[list[int]] = None,
                 num_transaction_sequences: int = 5,
                 max_seq_len: int = 2048,
                 num_timestamp_bins: int = 50,
                 num_amount_bins: int = 200,
                 cached: bool = False,
                 nrows: Optional[int] = None,
                 stride: int = 2,
                 return_labels: bool = False,
                 skip_agency_number: bool = False,
                 use_pretrained_tokenizer: bool = False,
    ):
        self.root = root
        self.fname = fname
        self.fextension = f'_{fextension}' if fextension else ''
        self.nrows = nrows
        self.cached = cached
        self.user_ids = user_ids
        self.return_labels = return_labels
        self.skip_agency_number = skip_agency_number

        self.trans_stride = stride

        self.stride = stride
        self.num_transaction_sequences = num_transaction_sequences

        self.max_seq_len = max_seq_len
        self.encoder_fit = {}

        self.trans_table = None
        self.data = []
        self.labels = []
        self.window_label = []

        self.tokenizer = NuTokenizer(model_name=model_name)

        self.ncols = None
        self.num_amount_bins = num_amount_bins
        self.num_timestamp_bins = num_timestamp_bins
        self.encode_data()
        if not use_pretrained_tokenizer:
            self.init_tokenizer()
        self.prepare_samples()
        self.save_tokenizer(vocab_dir)

    @classmethod
    def from_raw_data(
        cls,
        fname: str,
        root: str = "./data/",
        **kwargs
    ):
        df = pd.read_csv(path.join(root, f"{fname}.csv"))
        df = NuTable.clean_all_and_rename(df=df)
        df.to_csv(path.join(root, f"{fname}.cleaned.csv"), index=False)
        return cls(fname=f"{fname}.cleaned", root=root, **kwargs)

    def __getitem__(self, index):
            return_data = torch.tensor(self.data[index], dtype=torch.long)
            if self.return_labels:
                return_data = (return_data, torch.tensor(self.labels[index], dtype=torch.long))
            return return_data

    def __len__(self):
        return len(self.data)

    def token_count(self):
        return sum([len(sequence) for sequence in self.data])

    def get_summary(self, verbose: bool = False):
        info_dict = {
            "num_samples": len(self.data),
            "num_tokens": self.token_count(),
            "num_features": self.ncols,
            "features": self.trans_table.columns,
            "num_transaction_sequences": self.num_transaction_sequences,
            "max_seq_len": self.max_seq_len,
            "num_amount_bins": self.num_amount_bins,
            "num_timestamp_bins": self.num_timestamp_bins,
        }
        if verbose:
            from nugpt.utils import print_dataset_summary
            print_dataset_summary(info_dict)
    
    def save_tokenizer(self, vocab_dir: str):
        file_name = path.join(vocab_dir, f'tokenizer{self.fextension}.pkl')
        log.info(f"saving tokenizer at {file_name}")
        with open(file_name, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    @staticmethod
    def label_fit_transform(
        column: pd.Series,
        enc_type: Literal["label", "time"] = "label"
    ) -> tuple[object, pd.Series]:
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)

    @staticmethod
    def time_encoder(X: pd.Series) -> pd.DataFrame:
        d = pd.to_datetime(X).astype(int)
        return pd.DataFrame(d)

    @staticmethod
    def amount_encoder(X: pd.Series) -> pd.DataFrame:
        amt = X.astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
        return pd.DataFrame(amt)
    
    def _time_binning(self, data: np.ndarray):
        qtls = np.arange(0.0, 1.0 + 1 / self.num_timestamp_bins, 1 / self.num_timestamp_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

    def _quantize_time(self, inputs: np.ndarray, bin_edges: np.ndarray):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.num_timestamp_bins) - 1  # Clip edges
        return quant_inputs
    
    def _quantize(self, data: np.ndarray, num_bins: int, min_value: float = 0.01, max_value: float = 1e7):
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

    def format_trans(self, trans_lst: pd.Series, column_names: list[str]):
        trans_lst = list(divide_chunks(trans_lst, len(column_names)))
        user_token_ids = []

        for trans in trans_lst:
            transaction = {col: val for col, val in zip(column_names, trans)}
            token_ids = self.tokenizer.tokenize_transaction(transaction, column_names)
            user_token_ids.append(token_ids)

        return user_token_ids

    def prepare_samples(self):
        log.info("preparing user-level transaction sequences...")
        column_names = list(self.trans_table.columns)
        user_column = DATA_TYPE_MAPPING["index"]

        # Group transactions by user
        user_groups = self.trans_table.groupby(user_column)

        for user, user_transactions in tqdm.tqdm(user_groups):
            user_token_ids = []
            
            for _, transaction in user_transactions.iterrows():
                token_ids = self.tokenizer.tokenize_transaction(transaction.to_dict(), column_names)
                user_token_ids.append(token_ids)
            
            if len(user_token_ids) < self.num_transaction_sequences:
                log.info(f"User {user} has less than {self.num_transaction_sequences} transactions.")
                sequence = user_token_ids
                flattened_sequence = [token for transaction in sequence for token in transaction]
                self.data.append(flattened_sequence[:self.max_seq_len])
            else:
                for i in range(0, len(user_token_ids) - self.num_transaction_sequences + 1, self.stride):
                    sequence = user_token_ids[i:i + self.num_transaction_sequences]
                    flattened_sequence = [token for transaction in sequence for token in transaction]

                    # truncate when > max_seq_len
                    if len(flattened_sequence) > self.max_seq_len:
                        flattened_sequence = flattened_sequence[:self.max_seq_len]
                    
                    self.data.append(flattened_sequence)

        log.info(f"number of samples: {len(self.data)}")

    def init_tokenizer(self):
        column_names = list(self.trans_table.columns)

        for column in column_names:
            unique_values = self.trans_table[column].unique()
            if column in NuTable.get_numerical_columns():
                self.tokenizer.add_numerical_tokens(column, unique_values)
            elif column in NuTable.get_categorical_columns():
                self.tokenizer.add_categorical_tokens(column, unique_values)
            else:
                log.info(f"skipping text column {column}")

        log.info(f"total columns: {list(column_names)}")
        log.info(f"total vocabulary size: {self.tokenizer.get_vocab_size()}")

    def encode_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}{self.fextension}.encoded.csv'
        data_file = path.join(self.root, f"{self.fname}.csv")

        if self.cached and path.isfile(path.join(dirname, fname)):
            log.info(f"cached encoded data is read from {fname}")
            self.trans_table = self.get_csv(path.join(dirname, fname))
            encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
            self.encoder_fit = pickle.load(open(encoder_fname, "rb"))
            return

        data = self.get_csv(data_file)
        log.info(f"{data_file} is read.")

        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = data[column].fillna('Unknown')
            else:
                data[column] = data[column].fillna(0)

        log.info("timestamp fit transform")
        timestamp = self.time_encoder(data['Timestamp'])
        timestamp_fit, timestamp = self.label_fit_transform(timestamp, enc_type="time")
        self.encoder_fit['Timestamp'] = timestamp_fit
        data['Timestamp'] = timestamp

        data['Amount'] = self.amount_encoder(data['Amount'])

        for column in ['Amount', 'Timestamp']:
            if column in data.columns:
                coldata = np.array(data[column])
                if column == 'Timestamp':
                    data[column], bin_edges = self._quantize(coldata, num_bins=self.num_amount_bins)
                else:
                    bin_edges, _, _ = self._time_binning(coldata)
                    data[column] = self._quantize(coldata, num_bins=self.num_timestamp_bins)
                self.encoder_fit[f"{column}_bins"] = bin_edges

        self.trans_table = data

        log.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.trans_table, path.join(dirname, fname))

        encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
        log.info(f"writing cached encoder fit to {encoder_fname}")
        pickle.dump(self.encoder_fit, open(encoder_fname, "wb"))

    def get_csv(self, fname: str):
        data = pd.read_csv(fname, nrows=self.nrows)
        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data: pd.DataFrame, fname: str):
        log.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)
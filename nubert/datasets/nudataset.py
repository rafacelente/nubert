from typing import Optional, List
from os import path
import pandas as pd
import numpy as np
import tqdm
import logging

from scipy.ndimage import shift

import torch
from torch.utils.data.dataset import Dataset

from nubert.utils import divide_chunks, NuTable, DATA_TYPE_MAPPING
from nubert.tokenizer import NuTokenizer
from nubert.config import NubertPreTrainConfig, NUBERT_DEFAULT_CONFIG_FILE

logger = logging.getLogger(__name__)
log = logger

class NuDataset(Dataset):
    def __init__(self,
                 model_name: str = "distilbert/distilbert-base-uncased",
                 root: str = "./data/",
                 fname: str = "nudataset",
                 fextension: str = "",
                 num_transaction_sequences: int = 5,
                 max_seq_len: int = 2048,
                 nrows: Optional[int] = None,
                 stride: int = 2,
                 use_pretrained_tokenizer: bool = False,
                 randomize_column_order: bool = False,
    ):
        self.root = root
        self.fname = fname
        self.fextension = f'_{fextension}' if fextension else ''
        self.nrows = nrows
        self.stride = stride
        self.num_transaction_sequences = num_transaction_sequences
        self.max_seq_len = max_seq_len
        self.randomize_column_order = randomize_column_order
        self.encoder_fit = {}
        self.trans_table : pd.DataFrame | None = None
        self.data = []

        self.tokenizer = NuTokenizer(model_name=model_name)

        self.ncols = None
        self.read_data()
        if not use_pretrained_tokenizer:
            self.init_tokenizer()
        self.prepare_samples()

    @classmethod
    def from_config(cls, config: NubertPreTrainConfig):
        if not config.from_cleaned_data:
            return cls.from_raw_data(
                fname=config.file_name,
                root=config.dataset_path,
                filter_list=config.filter_list,
                num_bins=config.num_bins,
                columns_to_drop=config.columns_to_drop,
                agency_names_to_remove=config.agency_names_to_remove,
                num_transaction_sequences=config.num_transactions,
                max_seq_len=config.max_length,
                stride=config.stride,
                randomize_column_order=config.randomize_column_order,
                nrows=config.nrows
            )
        return cls.from_cleaned_data(
            model_name=config.model_name,
            root=config.dataset_path,
            fname=config.file_name,
            num_transaction_sequences=config.num_transactions,
            max_seq_len=config.max_length,
            stride=config.stride,
            randomize_column_order=config.randomize_column_order,
            nrows=config.nrows
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: str
    ):
        config = NubertPreTrainConfig.from_yaml(config_path)
        return cls.from_confg(config)

    @classmethod
    def from_default_config(cls):
        return cls.from_config_file(NUBERT_DEFAULT_CONFIG_FILE)

    @classmethod
    def from_raw_data(
        cls,
        fname: str,
        root: str = "./data/",
        filter_list: Optional[list[str]] = None,
        num_bins: int = 20,
        columns_to_drop: List[str] = ["Posted Date", "Year-Month", "Cardholder Last Name", "Cardholder First Initial", "Agency Number"],
        agency_names_to_remove: List[str] = ["EMPLOYEE BENEFITS"],
        **kwargs
    ):
        df = pd.read_csv(path.join(root, f"{fname}.csv"))
        log.info(f"cleaning and renaming columns in {fname}.csv")
        df = NuTable.clean_all_and_rename(
            df=df,
            num_bins=num_bins,
            columns_to_drop=columns_to_drop,
            agency_names_to_remove=agency_names_to_remove
        )
        if filter_list is not None and len(filter_list) > 0:
            df = NuTable.filter_to_list_of_agency_names(df, filter_list)
        df.to_csv(path.join(root, f"{fname}.cleaned.csv"), index=False)
        log.info(f"saved cleaned data to {fname}.cleaned.csv")
        return cls(fname=f"{fname}.cleaned", root=root, **kwargs)
    
    @classmethod
    def from_cleaned_data(
        cls,
        fname: str,
        root: str = "./data/",
        **kwargs
    ):
        nudataset = cls(fname=fname, root=root, **kwargs)
        warnings = NuTable.validate_cleaned_data(nudataset.trans_table)
        for warning in warnings:    
            log.warning(f"{warning}")
        return nudataset

    def __getitem__(self, index):
            return_data = torch.tensor(self.data[index], dtype=torch.long)
            return return_data

    def __len__(self):
        return len(self.data)

    def token_count(self):
        return sum([len(sequence) for sequence in self.data])

    def get_summary(self, verbose: bool = False):
        info_dict = {
            "num_samples": len(self.data),
            "num_tokens": self.token_count(),
            "num_features": self.trans_table.shape[1],
            "features": self.trans_table.columns,
            "num_transaction_sequences": self.num_transaction_sequences,
            "max_seq_len": self.max_seq_len,
        }
        if verbose:
            from nubert.utils import print_dataset_summary
            print_dataset_summary(info_dict)

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
        user_column = DATA_TYPE_MAPPING["index"]
        # Group transactions by user and filter < self.num_transactions
        self.trans_table = self.trans_table.groupby(user_column).filter(lambda x: len(x) >= self.num_transaction_sequences)
        self.trans_table = self.trans_table.sort_values(by=["Agency Name", "Transaction Date"]).reset_index(drop=True)
        user_groups = self.trans_table.groupby(user_column)

        for user, user_transactions in tqdm.tqdm(user_groups):
            grouper = np.arange(user_transactions.shape[0]) // self.num_transaction_sequences
            for _ in range(self.num_transaction_sequences):
                # we could probably cache the previous sequences to do
                # the shifting but j'ai la flemme
                for group_id, group in user_transactions.groupby(grouper):
                    if group_id == -1:
                        continue
                    flattened_sequence = []
                    for _, (_, transaction) in enumerate(group.iterrows()):
                        column_order = ['Agency Name', 'Vendor', 'Merchant Category Code (MCC)', 'Timestamp', 'Amount']
                        if self.randomize_column_order:
                            np.random.shuffle(column_order)
                        flattened_sequence.extend(
                            self.tokenizer.tokenize_transaction(
                                transaction.to_dict(),
                                column_order=column_order
                            )
                        )
                        if len(flattened_sequence) > self.max_seq_len:
                                flattened_sequence = flattened_sequence[:self.max_seq_len]
                    self.data.append(flattened_sequence)
                grouper = shift(grouper, self.stride, cval=-1)

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

    def read_data(self):
        data_file = path.join(self.root, f"{self.fname}.csv")
        data = self.get_csv(data_file)
        log.info(f"{data_file} is read.")
        self.trans_table = data

    def get_csv(self, fname: str):
        data = pd.read_csv(fname, nrows=self.nrows)
        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data: pd.DataFrame, fname: str):
        log.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)
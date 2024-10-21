from typing import Optional, List
from os import path
import numpy as np
import pandas as pd
import tqdm
import logging

import torch
from scipy.ndimage import shift

from nubert.utils import divide_chunks, NuTable, DATA_TYPE_MAPPING
from nubert.datasets import NuDataset
from nubert.config import AmountConfig, AMOUNT_DEFAULT_CONFIG_FILE

logger = logging.getLogger(__name__)
log = logger

class AmountDataset(NuDataset):
    def __init__(self, prediction_column: str = "Amount", *args, **kwargs):
        self.prediction_column = prediction_column
        self.labels = []
        super().__init__(use_pretrained_tokenizer=True, *args, **kwargs)

    @classmethod
    def from_config(cls, config: AmountConfig):
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
        config = AmountConfig.from_yaml(config_path)
        return cls.from_config(config)

    @classmethod
    def from_default_config(cls):
        return cls.from_config_file(AMOUNT_DEFAULT_CONFIG_FILE)

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
        df = NuTable.clean_all_and_rename_amount(
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

    def __getitem__(self, index):
            input_ids = self.data[index]
            attention_mask = [1] * len(input_ids)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "label": torch.tensor(self.labels[index], dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    def __len__(self):
        return len(self.data)

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
            logging.info(f"User: {user} | Number of transactions: {user_transactions.shape[0]}")
            for _ in range(self.num_transaction_sequences):
                # we could probably cache the previous sequences to do
                # the shifting but j'ai la flemme
                for group_id, group in user_transactions.groupby(grouper):
                    if group_id == -1:
                        continue
                    flattened_sequence = []
                    for sequence_i, (table_i, transaction) in enumerate(group.iterrows()):
                        if sequence_i == self.num_transaction_sequences - 1:
                            self.trans_table.loc[table_i, 'Prediction'] = transaction[self.prediction_column]
                            self.labels.append(transaction[self.prediction_column])
                            flattened_sequence.extend(self.tokenizer.tokenize_transaction_with_mask_amount(transaction.to_dict(), column_order=['Agency Name', 'Vendor', 'Merchant Category Code (MCC)', 'Timestamp', 'Amount']))
                            # truncate when > max_seq_len
                            if len(flattened_sequence) > self.max_seq_len:
                                flattened_sequence = flattened_sequence[:self.max_seq_len]
                            self.data.append(flattened_sequence)
                        else:
                            flattened_sequence.extend(self.tokenizer.tokenize_transaction(transaction.to_dict(), column_order=['Agency Name', 'Vendor', 'Merchant Category Code (MCC)', 'Timestamp', 'Amount']))

                    assert len(self.data) == len(self.labels)
                grouper = shift(grouper, self.stride, cval=-1)

        log.info(f"number of samples: {len(self.data)}")
from os import path
import numpy as np
import pandas as pd
import tqdm
import logging


import torch

from nugpt.utils import divide_chunks
from nugpt.utils import DATA_TYPE_MAPPING
from nugpt.datasets import NuDataset

logger = logging.getLogger(__name__)
log = logger

class AmountDataset(NuDataset):
    def __init__(self, prediction_column: str = "Amount", *args, **kwargs):
        self.prediction_column = prediction_column
        self.labels = []
        super().__init__(use_pretrained_tokenizer=True, *args, **kwargs)

    def __getitem__(self, index):
            return {
                "text": torch.tensor(self.data[index], dtype=torch.long),
                "label": torch.tensor(self.labels[index], dtype=torch.long)
            }

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def amount_encoder(X: pd.Series) -> pd.DataFrame:
        amt = X.apply(lambda x: x.strip("$()").replace(",", "")).astype(float)
        return pd.DataFrame(amt)


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

        # Group transactions by user and filter < self.num_transactions
        self.trans_table = self.trans_table.groupby(user_column).filter(lambda x: len(x) >= self.num_transaction_sequences)
        user_groups = self.trans_table.groupby(user_column)

        for user, user_transactions in user_groups:
            user_transactions.sort_values("Timestamp", inplace=True)
            grouper = np.arange(user_transactions.shape[0]) // self.num_transaction_sequences
            for _, group in user_transactions.groupby(grouper):
                flattened_sequence = []
                for sequence_i, (_, transaction) in enumerate(group.iterrows()):
                    if sequence_i == self.num_transaction_sequences - 1:
                        self.labels.append(transaction['Amount'])
                        flattened_sequence.extend(self.tokenizer.tokenize_transaction_with_mask_amount(transaction.to_dict(), column_order=['AgencyName', 'Vendor', 'MCC', 'Timestamp', 'Amount']))
                    else:
                        flattened_sequence.extend(self.tokenizer.tokenize_transaction(transaction.to_dict(), column_order=['AgencyName', 'Vendor', 'MCC', 'Timestamp', 'Amount']))

                # Truncate when > max_seq_len
                if len(flattened_sequence) > self.max_seq_len:
                    flattened_sequence = flattened_sequence[:self.max_seq_len]

                self.data.append(flattened_sequence)

        log.info(f"number of samples: {len(self.data)}")
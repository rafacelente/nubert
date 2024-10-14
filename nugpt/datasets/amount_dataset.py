from os import path
import math
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
        amt = X.apply(lambda x: x.strip("$()").replace(",", "")).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
        return pd.DataFrame(amt)


    def format_trans(self, trans_lst: pd.Series, column_names: list[str]):
        trans_lst = list(divide_chunks(trans_lst, len(column_names)))
        user_token_ids = []

        column_names = [col for col in column_names if col != self.prediction_column]

        for trans in trans_lst:
            transaction = {col: val for col, val in zip(column_names, trans)}
            token_ids = self.tokenizer.tokenize_transaction(transaction, column_names)
            user_token_ids.append(token_ids)

        return user_token_ids
    
    def prepare_samples(self):
        log.info("preparing user-level transaction sequences...")
        column_names = [column for column in list(self.trans_table.columns) if column != self.prediction_column]
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
                self.labels.append(transaction[self.prediction_column])
            else:
                for i in range(0, len(user_token_ids) - self.num_transaction_sequences + 1, self.stride):
                    sequence = user_token_ids[i:i + self.num_transaction_sequences]
                    flattened_sequence = [token for transaction in sequence for token in transaction]

                    # truncate when > max_seq_len
                    if len(flattened_sequence) > self.max_seq_len:
                        flattened_sequence = flattened_sequence[:self.max_seq_len]
                    
                    self.data.append(flattened_sequence)
                    self.labels.append(transaction[self.prediction_column])

        log.info(f"number of samples: {len(self.data)}")

    
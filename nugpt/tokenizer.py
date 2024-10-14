from transformers import AutoTokenizer
from typing import List, Dict, Any, Union
import numpy as np
from nugpt.utils import INVERSE_RENAME_MAPPING

class NuTokenizer:
    def __init__(
            self,
            model_name: str = "TinyLlama/TinyLlama_v1.1"
        ):
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.special_tokens = {
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
            "cls_token": "[CLS]",
        }
        self.add_special_tokens()
        self.categorical_encoders = {}
        self.numerical_encoders = {}


    def add_special_tokens(self):
        special_tokens_dict = {k: v for k, v in self.special_tokens.items() if v not in self.base_tokenizer.vocab}
        self.base_tokenizer.add_special_tokens(special_tokens_dict)
        
    def add_categorical_tokens(self, column: str, unique_values: List[Any]):
        self.categorical_encoders[column] = {value: f"<{column}_{value}>" for value in unique_values}
        new_tokens = list(self.categorical_encoders[column].values())
        self.base_tokenizer.add_tokens(new_tokens)

    def add_numerical_tokens(self, column: str, bins: List[Union[float, int]]):
        self.numerical_encoders[column] = {value: f"<{column}_bin_{value}>" for i, value in enumerate(bins)}
        new_tokens = list(self.numerical_encoders[column].values())
        self.base_tokenizer.add_tokens(new_tokens)
        
    def encode_categorical(self, column: str, value: Any) -> str:
        return self.categorical_encoders[column].get(value, self.special_tokens["unk_token"])
    
    def encode_numerical(self, column: str, value: float, bins: np.ndarray) -> str:
        return self.numerical_encoders[column][value]
    
    def tokenize_transaction_with_mask_amount(
            self,
            transaction: Dict[str, Any],
            column_order: List[str]
        ) -> List[int]:
        # [CLS] field1: value1 [SEP] field2: value2 [SEP] ... fieldN: valueN [SEP] [SEP]
        tokens = [self.special_tokens["cls_token"]]
        
        for column in column_order:
            index_tokens = self.base_tokenizer.tokenize(f"{INVERSE_RENAME_MAPPING[column]}:")
            tokens.extend(index_tokens)
            value = transaction[column]
            if column in self.categorical_encoders:
                token = self.encode_categorical(column, value)
            elif column in self.numerical_encoders:
                bins = list(self.numerical_encoders[f"{column}"].keys())
                token = self.encode_numerical(column, value, bins)
            elif column == "Amount":
                tokens.append(self.base_tokenizer.mask_token)
                tokens.append(self.special_tokens["sep_token"])
                continue
            else:
                # For text fields, use the base tokenizer
                subtokens = self.base_tokenizer.tokenize(str(value))
                tokens.extend(subtokens)
                tokens.append(self.special_tokens["sep_token"])
                continue
            
            tokens.append(token)
            tokens.append(self.special_tokens["sep_token"])
        
        tokens.append(self.special_tokens["sep_token"]) 
        return self.base_tokenizer.convert_tokens_to_ids(tokens)
    
    def tokenize_transaction(self, transaction: Dict[str, Any], column_order: List[str]) -> List[int]:
        # [CLS] field1: value1 [SEP] field2: value2 [SEP] ... fieldN: valueN [SEP] [SEP]
        tokens = [self.special_tokens["cls_token"]]
        
        for column in column_order:
            index_tokens = self.base_tokenizer.tokenize(f"{INVERSE_RENAME_MAPPING[column]}:")
            tokens.extend(index_tokens)
            value = transaction[column]
            if column in self.categorical_encoders:
                token = self.encode_categorical(column, value)
            elif column in self.numerical_encoders:
                bins = list(self.numerical_encoders[f"{column}"].keys())
                token = self.encode_numerical(column, value, bins)
            else:
                # For text fields, use the base tokenizer
                subtokens = self.base_tokenizer.tokenize(str(value))
                tokens.extend(subtokens)
                tokens.append(self.special_tokens["sep_token"])
                continue
            
            tokens.append(token)
            tokens.append(self.special_tokens["sep_token"])
        
        tokens.append(self.special_tokens["sep_token"]) 
        return self.base_tokenizer.convert_tokens_to_ids(tokens)
    
    def tokenize_sequence(self, transactions: List[Dict[str, Any]], column_order: List[str]) -> List[int]:
        # [CLS] transaction1_field1: value1 [SEP] ... transaction1_fieldN: valueN [SEP] [SEP] [CLS] transaction2_field1: value1 [SEP] ... transaction2_fieldN: valueN [SEP] [SEP] ...
        sequence_tokens = []
        for transaction in transactions:
            sequence_tokens.extend(self.tokenize_transaction(transaction, column_order))
        return sequence_tokens
    
    def decode(self, token_ids: List[int]) -> str:
        return self.base_tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        return len(self.base_tokenizer)
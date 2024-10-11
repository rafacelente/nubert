from typing import Union
from collections import OrderedDict
import numpy as np
import torch

class Vocabulary:
    def __init__(
            self,
            adap_threshold: int = 10000,
        ):
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"


        self.adap_thres = adap_threshold
        self.adap_sm_cols = set()

        self.special_field_tag = "SPECIAL"

        self.special_tokens = [self.unk_token, self.sep_token, self.pad_token,
                               self.cls_token, self.mask_token, self.bos_token, self.eos_token]
        
        self.token2id = OrderedDict()  # {field: {token: id}, ...}
        self.id2token = OrderedDict()  # {id : [token,field]}
        self.field_keys = OrderedDict()
        self.token2id[self.special_field_tag] = OrderedDict()

        for token in self.special_tokens:
            self.set_id(token, self.special_field_tag)

    def set_id(self, token: str, field_name: str, return_local: bool = False):
        global_id, local_id = None, None

        if field_name not in self.token2id:
            if not len(self.token2id):
                raise ValueError("Field keys not set. Use set_field_keys to set field keys")
            raise ValueError(f"Field name {field_name} not found in vocabulary. Fields are {self.token2id.keys()}")

        if token not in self.token2id[field_name]:
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])
            self.token2id[field_name][token] = [global_id, local_id]
            self.id2token[global_id] = [token, field_name, local_id]
        else:
            global_id, local_id = self.token2id[field_name][token]

        if return_local:
            return local_id
        return global_id

    def get_id(self, token: str, field_name: str = "", special_token: bool = False, return_local: bool = False):
        if special_token:
            field_name = self.special_field_tag
        if field_name not in self.token2id:
            raise ValueError(f"Field name {field_name} not found in vocabulary. Fields are {self.token2id.keys()}")
        return self.token2id[field_name][token][int(return_local)]

    def get_tokens(self, field_name: str):
        return self.token2id[field_name].keys()

    def set_field_keys(self, keys: list[str]):
        for key in keys:
            self.token2id[key] = OrderedDict()
            self.field_keys[key] = len(self.field_keys)
        self.field_keys[self.special_field_tag] = len(self.field_keys)

    def get_field_ids(self, field_name: str, return_local: bool = False):
        return [self.token2id[field_name][token][int(return_local)] for token in self.token2id[field_name]]
    
    def get_local_from_global_ids(self, global_ids: Union[list[int], torch.Tensor]) -> np.ndarray:
        if isinstance(global_ids, torch.Tensor):
            global_ids = global_ids.tolist()
        def map_global_to_local(id):
            return self.id2token[id][2] if id != -100 else -100
        return np.vectorize(map_global_to_local)(global_ids)
    
    def get_special_tokens(self):
        special_tokens_map = {}
        # TODO : remove the dependency of re-initializing here. retrieve from field_key = SPECIAL
        keys = ["unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
        for key, token in zip(keys, self.special_tokens):
            token = "%s_%s" % (self.special_field_tag, token)
            special_tokens_map[key] = token
        return special_tokens_map

    def save_vocab(self, fname):
        self.filename = fname
        with open(fname, "w") as fout:
            for idx in self.id2token:
                token, field, _ = self.id2token[idx]
                token = f"{token}_{field}"
                fout.write("%s\n" % token)

    def get_field_keys(self, ignore_special=False):
        keys = list(self.field_keys.keys())

        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def __len__(self):
        return len(self.id2token)

    def __str__(self):
        str_ = 'vocab: [{} tokens]  [field_keys={}]'.format(len(self), self.field_keys)
        return str_
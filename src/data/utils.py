from hydra import initialize, compose
try:
    initialize('../')
except ValueError:
    pass
cfg = compose(config_name='config')
import pandas as pd
import numpy as np
import random
import torch
from transformers import AutoTokenizer
import re
from src.PATHS import *
from torch.utils.data import Dataset, DataLoader
import shutil
 
def get_labels(raw_labels, icd_dict):
    icd_list = str(raw_labels).split(";")
    positive = [icd_dict.get(icd) for icd in icd_list]
    labels = np.zeros(len(icd_dict))
    try:
        labels[positive] = 1.0
    except:
        pass
    return list(labels)

def set_seed(seed: int = cfg.seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model, do_lower_case=True)
def tokenize_function(rows):
    if cfg.model == "PLM-ICD":
        tokenizer_dict = tokenizer(rows["TEXT"], max_length=cfg.n_tokens, padding=False, truncation=True, add_special_tokens="cls" not in cfg.model_mode)
        input_ids = tokenizer_dict["input_ids"]
        attn_mask = tokenizer_dict["attention_mask"]

        return input_ids, attn_mask

    elif cfg.model == "HiBERT":
        tokenizer_dict = tokenizer(rows["TEXT"], max_length=cfg.n_tokens, padding=False, truncation=True)
        input_ids = tokenizer_dict["input_ids"]
        attn_mask = tokenizer_dict["attention_mask"]
        return input_ids, attn_mask

def hibert_collate_fn(x):
    words = [x_['input_ids'] for x_ in x]
    masks = [x_['attention_mask'] for x_ in x]
    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)    # max of batch

    input_ids = pad_sequence(words, max_seq_len)
    masks = pad_sequence(masks, max_seq_len)
    labels = [x_['LABEL'] for x_ in x]

    return torch.tensor(input_ids), torch.tensor(masks), torch.tensor(labels)

def pad_sequence(x, max_len, type=np.int):
    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row
    return padded_x

def plmicd_collate_fn(x):
    if "cls" in cfg.model_mode:
        for f in x:
            new_input_ids = []
            for i in range(0, len(f["input_ids"]), cfg.chunk_size - 2):
                new_input_ids.extend([tokenizer.cls_token_id] + f["input_ids"][i:i+(cfg.chunk_size)-2] + [tokenizer.sep_token_id])
            f["input_ids"] = new_input_ids
            f["attention_mask"] = [1] * len(f["input_ids"])

    max_length = max([len(f["input_ids"]) for f in x])
    if max_length % cfg.chunk_size != 0:
        max_length = max_length - (max_length % cfg.chunk_size) + cfg.chunk_size

    input_ids = torch.tensor([
        f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
        for f in x
    ]).contiguous().view((len(x), -1, cfg.chunk_size))

    masks = torch.tensor([
        f["attention_mask"] + [0] * (max_length - len(f["attention_mask"]))
        for f in x
    ]).contiguous().view((len(x), -1, cfg.chunk_size))

    labels = torch.tensor([x_['LABEL'] for x_ in x])

    return input_ids, masks, labels

from hydra import initialize, compose
try:
    initialize('../')
except ValueError:
    pass
cfg = compose(config_name='config')
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .utils import tokenize_function, get_labels, hibert_collate_fn, plmicd_collate_fn
from ..PATHS import *
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model, do_lower_case=True)

class TextDataset(Dataset):
    def __init__(self, df, transforms=None):
        
        self.df = df
        self.transforms=transforms

    def __len__(self):
        """
        This is simply the number of labels in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Generate one sample of data
        """
        return self.df.iloc[idx]

def prepare_train_loaders(mode=cfg.dataset):
    if mode=="full":
        train_df = pd.read_csv(PROCESSED_DATA_PATH / "train_full.csv").reset_index(drop=True)
        val_df = pd.read_csv(PROCESSED_DATA_PATH / "dev_full.csv").reset_index(drop=True)
        icd_dict = pd.read_csv(PROCESSED_DATA_PATH / "FULL_CODES.csv", header=None).squeeze().to_dict()
        icd_dict = {v:k for k, v in icd_dict.items()}
    else:
        train_df = pd.read_csv(PROCESSED_DATA_PATH / f"train_{mode}.csv").reset_index(drop=True)
        val_df = pd.read_csv(PROCESSED_DATA_PATH / f"dev_{mode}.csv").reset_index(drop=True)
        icd_dict = pd.read_csv(PROCESSED_DATA_PATH / f"TOP_{mode}_CODES.csv", header=None).squeeze().to_dict()
        icd_dict = {v:k for k, v in icd_dict.items()}

    train_df["LABEL"] = train_df["LABELS"].apply(get_labels, icd_dict=icd_dict)
    val_df["LABEL"] = val_df["LABELS"].apply(get_labels, icd_dict=icd_dict)

    train_df[["input_ids", "attention_mask"]] = train_df.apply(tokenize_function, axis=1, result_type="expand")
    val_df[["input_ids", "attention_mask"]] = val_df.apply(tokenize_function, axis=1, result_type="expand")

    train_ds = TextDataset(train_df[["LABEL", "input_ids", "attention_mask"]])
    val_ds = TextDataset(val_df[["LABEL", "input_ids", "attention_mask"]])

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=cfg.train_batch_size, drop_last=False, collate_fn=hibert_collate_fn if cfg.model=="HiBERT" else plmicd_collate_fn)
    val_dataloader = DataLoader(val_ds, shuffle=False, batch_size=cfg.val_batch_size, drop_last=False, collate_fn=hibert_collate_fn if cfg.model=="HiBERT" else plmicd_collate_fn)

    return train_dataloader, val_dataloader

def prepare_val_loader(mode=cfg.dataset):

    if mode=="full":
        val_df = pd.read_csv(PROCESSED_DATA_PATH / "dev_full.csv").reset_index(drop=True)
        icd_dict = pd.read_csv(PROCESSED_DATA_PATH / "FULL_CODES.csv", header=None).squeeze().to_dict()
        icd_dict = {v:k for k, v in icd_dict.items()}
    else:
        val_df = pd.read_csv(PROCESSED_DATA_PATH / f"dev_{mode}.csv").reset_index(drop=True)
        icd_dict = pd.read_csv(PROCESSED_DATA_PATH / f"TOP_{mode}_CODES.csv", header=None).squeeze().to_dict()
        icd_dict = {v:k for k, v in icd_dict.items()}

    val_df["LABEL"] = val_df["LABELS"].apply(get_labels, icd_dict=icd_dict)

    val_df[["input_ids", "attention_mask"]] = val_df.apply(tokenize_function, axis=1, result_type="expand")

    val_ds = TextDataset(val_df[["LABEL", "input_ids", "attention_mask"]])
    val_dataloader = DataLoader(val_ds, shuffle=False, batch_size=cfg.val_batch_size, drop_last=False, collate_fn=hibert_collate_fn if cfg.model=="HiBERT" else plmicd_collate_fn)

    return val_dataloader

def prepare_test_loader(mode=cfg.dataset):
    if mode=="full":
        test_df = pd.read_csv(PROCESSED_DATA_PATH / "test_full.csv").reset_index(drop=True)
        icd_dict = pd.read_csv(PROCESSED_DATA_PATH / "FULL_CODES.csv", header=None).squeeze().to_dict()
        icd_dict = {v:k for k, v in icd_dict.items()}
    else:
        test_df = pd.read_csv(PROCESSED_DATA_PATH / f"test_{mode}.csv").reset_index(drop=True)
        icd_dict = pd.read_csv(PROCESSED_DATA_PATH / f"TOP_{mode}_CODES.csv", header=None).squeeze().to_dict()
        icd_dict = {v:k for k, v in icd_dict.items()}
    
    test_df["LABEL"] = test_df["LABELS"].apply(get_labels, icd_dict=icd_dict)

    test_df[["input_ids", "attention_mask"]] = test_df.apply(tokenize_function, axis=1, result_type="expand")

    test_ds = TextDataset(test_df[["LABEL", "input_ids", "attention_mask"]])
    test_dataloader = DataLoader(test_ds, shuffle=False, batch_size=cfg.val_batch_size, drop_last=False, collate_fn=hibert_collate_fn if cfg.model=="HiBERT" else plmicd_collate_fn)

    return test_dataloader



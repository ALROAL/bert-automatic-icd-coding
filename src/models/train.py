from hydra import initialize, compose
try:
    initialize('../')
except ValueError:
    pass
cfg = compose(config_name='config')
import pandas as pd
import numpy as np
import torch
import transformers
from torch.cuda import amp
import copy
from .models import build_model, save_model
from src.data.dataloaders import prepare_train_loaders, prepare_test_loader
from src.data.utils import set_seed
from .evaluate import all_metrics
from src.PATHS import *
import wandb
import collections

#Loss
def get_loss():
    if cfg.weighted_loss:
        if cfg.dataset =="full":
            train_df = pd.read_csv(PROCESSED_DATA_PATH / "train_full.csv").reset_index(drop=True)
            icd_codes = pd.read_csv(PROCESSED_DATA_PATH / "FULL_CODES.csv", header=None).squeeze().to_dict()

            icd_dict = pd.read_csv(PROCESSED_DATA_PATH / "FULL_CODES.csv", header=None).squeeze().to_dict()
            icd_dict = {v:k for k, v in icd_dict.items()}
        else:
            train_df = pd.read_csv(PROCESSED_DATA_PATH / f"train_{cfg.dataset}.csv").reset_index(drop=True)
            icd_codes = pd.read_csv(PROCESSED_DATA_PATH / f"TOP_{cfg.dataset}_CODES.csv", header=None).squeeze().to_dict()

        train_icd_codes = [icd_code for icd_list in train_df["LABELS"].values for icd_code in str(icd_list).split(";")]
        n = len(train_df)
        counts = collections.Counter(train_icd_codes)
        pos_weight = []
        for icd in icd_codes.values():
            try:
                c = counts[icd]
                w = (n-c)/c
                pos_weight.append(w)
            except:
                pos_weight.append(1)
        pos_weight = np.array(pos_weight)
        #pos_weight = np.log(pos_weight) #mild-water-21
        #pos_weight = pos_weight / pos_weight.max() #sage-sea
        pos_weight = (pos_weight - pos_weight.min())/(pos_weight.max()-pos_weight.min()) + 1
        pos_weight = torch.tensor(pos_weight).to(cfg.device)
    else:
        pos_weight=None

    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#Optimizer
def get_optimizer(model):
   
    parameters = model.parameters()
    
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=cfg.lr)
    
    elif cfg.optimizer == 'AdamW':
        if cfg.model == "PLM-ICD":
            no_decay = ["bias", "LayerNorm.weight"]
            parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }]

            optimizer = transformers.AdamW(parameters, lr=cfg.lr)

        else:
            optimizer = transformers.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay, correct_bias=False)
        
    return optimizer

#Scheduler
def get_scheduler(optimizer, dataloader):
    
    T_max = int(cfg.epochs/6)

    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=5e-6)

    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.01, min_lr=1e-6)

    elif cfg.scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    elif cfg.scheduler == 'LinearWarmup':
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, cfg.warm_up_epochs, cfg.epochs)

    elif cfg.scheduler == "ConstantWarmup":
        scheduler = [transformers.get_constant_schedule_with_warmup(optimizer, cfg.warm_up_epochs), 
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.01, min_lr=1e-6)]
        
    return scheduler

#Train functions
def train_one_epoch(model, dataloader, criterion, optimizer):

    model.train()
    scaler = amp.GradScaler()

    loss_sum = 0
    n_samples = 0

    for step, (input_ids, attention_mask, labels) in enumerate(dataloader):

        input_ids = input_ids.to(cfg.device)
        attention_mask = attention_mask.to(cfg.device)
        labels = labels.to(cfg.device)

        batch_size = labels.size(0)
        n_samples += batch_size

        with amp.autocast(enabled=True):
            y_pred = model(input_ids, attention_mask)
            loss = criterion(y_pred, labels)
            loss_sum += loss.item()*batch_size
            loss = loss / cfg.n_accumulate

        scaler.scale(loss).backward()
        if (step + 1) % cfg.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()

    epoch_loss = loss_sum / n_samples
    torch.cuda.empty_cache()
    
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion):

    model.eval()
    loss_sum = 0
    n_samples = 0

    y_pred_final = np.array([])
    y_pred_raw_final = np.array([])
    labels_final = np.array([])
    for input_ids, attention_mask, labels in dataloader:

        input_ids = input_ids.to(cfg.device)
        attention_mask = attention_mask.to(cfg.device)
        labels = labels.to(cfg.device)

        batch_size = labels.size(0)
        n_samples += batch_size
        
        y_pred = model(input_ids, attention_mask)

        loss = criterion(y_pred, labels)
        loss_sum += loss.item()*batch_size

        pred_raw = torch.nn.Sigmoid()(y_pred)
        y_pred = (pred_raw>0.5).to(dtype=torch.float).cpu().detach().numpy()
        pred_raw = pred_raw.cpu().detach().numpy()
        labels = labels.to(dtype=torch.float).cpu().detach().numpy()

        y_pred_final = np.concatenate((y_pred_final, y_pred)) if y_pred_final.size else y_pred
        y_pred_raw_final = np.concatenate((y_pred_raw_final, pred_raw)) if y_pred_raw_final.size else pred_raw
        labels_final = np.concatenate((labels_final, labels)) if labels_final.size else labels

    metrics_dict = all_metrics(y_pred_final, labels_final, k=5, yhat_raw=y_pred_raw_final, calc_auc=True)

    epoch_loss = loss_sum / n_samples
    torch.cuda.empty_cache()
    
    return epoch_loss, metrics_dict

def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler):

    best_criterion = 0
    for epoch in range(cfg.epochs):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, metrics_dict = valid_one_epoch(model, val_loader, criterion)

        # deep copy the model weights
        if metrics_dict[cfg.criterion] > best_criterion:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_criterion = metrics_dict[cfg.criterion]

        logging_dict = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            cfg.criterion: metrics_dict[cfg.criterion]}
        logging_dict["lr"] = optimizer.param_groups[0]['lr']
        wandb.log(logging_dict)

        if scheduler is not None:
            scheduler.step()

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

def train():
    
    set_seed()
    model = build_model()
    train_loader, val_loader = prepare_train_loaders()
    criterion = get_loss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, train_loader)

    wandb.init(project="test-project", entity="icd_classification", config=cfg._content, job_type='train')

    model = run_training(model, train_loader, val_loader, criterion, optimizer, scheduler)
    model_name = f'model_{cfg.dataset}.pth'
    save_model(model, model_name)

    val_loss, val_metrics_dict= valid_one_epoch(model, val_loader, criterion)
    wandb.run.summary["val_loss"] = val_loss
    for k, v in val_metrics_dict.items():
        wandb.run.summary[f"val_{k}"] = v

    test_loader = prepare_test_loader()
    test_loss, test_metrics_dict = valid_one_epoch(model, test_loader, criterion)
    wandb.run.summary["test_loss"] = test_loss
    for k, v in test_metrics_dict.items():
        wandb.run.summary[f"test_{k}"] = v

    

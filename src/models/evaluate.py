from hydra import initialize, compose
try:
    initialize('../')
except ValueError:
    pass
cfg = compose(config_name='config')
import numpy as np
import torch
from tqdm import tqdm
import csv
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from .models import load_model
from src.data.dataloaders import prepare_val_loader, prepare_test_loader
from src.PATHS import MODELS_PATH


@torch.no_grad()
def evaluate_model(model=None, split="val", mode=cfg.dataset, device=cfg.device):

    if model is not None:
        model.eval()
    else:
        model = load_model()

    if split=="val":
        dataloader = prepare_val_loader(mode)
    else:
        dataloader = prepare_test_loader(mode)

    model.eval()
    y_pred_final = np.array([])
    y_pred_raw_final = np.array([])
    labels_final = np.array([])
    for input_ids, attention_mask, labels in dataloader: 
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        y_pred = model(input_ids, attention_mask)
        pred_raw = torch.nn.Sigmoid()(y_pred)
        y_pred = (pred_raw>0.5).to(dtype=torch.float).cpu().detach().numpy()
        pred_raw = pred_raw.cpu().detach().numpy()
        labels = labels.to(dtype=torch.float).detach().numpy()

        y_pred_final = np.concatenate((y_pred_final, y_pred)) if y_pred_final.size else y_pred
        y_pred_raw_final = np.concatenate((y_pred_raw_final, pred_raw)) if y_pred_raw_final.size else pred_raw
        labels_final = np.concatenate((labels_final, labels)) if labels_final.size else labels
    
    metrics_dict = all_metrics(y_pred_final, labels_final, k=5, yhat_raw=y_pred_raw_final, calc_auc=True)
    torch.cuda.empty_cache()
    
    return metrics_dict

# Code from https://agit.ai/jsx/MCA_BERT/src/branch/master/mc_ptm/utils/evaluate.py
def print_metrics(metrics):
    print()
    if "auc_macro" in metrics.keys():
        print("[MACRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"], metrics["auc_macro"]))
    else:
        print("[MACRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"]))

    if "auc_micro" in metrics.keys():
        print("[MICRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"]))
    else:
        print("[MICRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"]))
    for metric, val in metrics.items():
        if metric.find("rec_at") != -1:
            print("%s: %.4f" % (metric, val))
    print()

def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

########################
# METRICS BY CODE TYPE
########################

def results_by_type(mdir, test_file):
    d2ind = {}
    p2ind = {}

    # get predictions for diagnoses and procedures
    diag_preds = defaultdict(lambda: set([]))
    proc_preds = defaultdict(lambda: set([]))
    preds = defaultdict(lambda: set())
    with open('%s/preds_test.psv' % mdir, 'r') as f:
        r = csv.reader(f, delimiter='|')
        for row in r:
            if len(row) > 1:
                for code in row[1:]:
                    preds[row[0]].add(code)
                    if code != '':
                        try:
                            pos = code.index('.')
                            if pos == 3 or (code[0] == 'E' and pos == 4):
                                if code not in d2ind:
                                    d2ind[code] = len(d2ind)
                                diag_preds[row[0]].add(code)
                            elif pos == 2:
                                if code not in p2ind:
                                    p2ind[code] = len(p2ind)
                                proc_preds[row[0]].add(code)
                        except:
                            if len(code) == 3 or (code[0] == 'E' and len(code) == 4):
                                if code not in d2ind:
                                    d2ind[code] = len(d2ind)
                                diag_preds[row[0]].add(code)
    # get ground truth for diagnoses and procedures
    diag_golds = defaultdict(lambda: set([]))
    proc_golds = defaultdict(lambda: set([]))
    golds = defaultdict(lambda: set())
    with open(test_file, 'r') as f:
        r = csv.reader(f)
        # header
        next(r)
        for row in r:
            codes = set([c for c in row[3].split(';')])
            for code in codes:
                golds[row[1]].add(code)
                try:
                    pos = code.index('.')
                    if pos == 3:
                        if code not in d2ind:
                            d2ind[code] = len(d2ind)
                        diag_golds[row[1]].add(code)
                    elif pos == 2:
                        if code not in p2ind:
                            p2ind[code] = len(p2ind)
                        proc_golds[row[1]].add(code)
                except:
                    if len(code) == 3 or (code[0] == 'E' and len(code) == 4):
                        if code not in d2ind:
                            d2ind[code] = len(d2ind)
                        diag_golds[row[1]].add(code)

    hadm_ids = sorted(set(diag_golds.keys()).intersection(set(diag_preds.keys())))

    ind2d = {i: d for d, i in d2ind.items()}
    ind2p = {i: p for p, i in p2ind.items()}
    type_dicts = (ind2d, ind2p)
    return diag_preds, diag_golds, proc_preds, proc_golds, golds, preds, hadm_ids, type_dicts
    

def diag_f1(diag_preds, diag_golds, ind2d, hadm_ids):
    num_labels = len(ind2d)
    yhat_diag = np.zeros((len(hadm_ids), num_labels))
    y_diag = np.zeros((len(hadm_ids), num_labels))
    for i, hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_diag_inds = [1 if ind2d[j] in diag_preds[hadm_id] else 0 for j in range(num_labels)]
        gold_diag_inds = [1 if ind2d[j] in diag_golds[hadm_id] else 0 for j in range(num_labels)]
        yhat_diag[i] = yhat_diag_inds
        y_diag[i] = gold_diag_inds
    return micro_f1(yhat_diag.ravel(), y_diag.ravel())


def proc_f1(proc_preds, proc_golds, ind2p, hadm_ids):
    num_labels = len(ind2p)
    yhat_proc = np.zeros((len(hadm_ids), num_labels))
    y_proc = np.zeros((len(hadm_ids), num_labels))
    for i, hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_proc_inds = [1 if ind2p[j] in proc_preds[hadm_id] else 0 for j in range(num_labels)]
        gold_proc_inds = [1 if ind2p[j] in proc_golds[hadm_id] else 0 for j in range(num_labels)]
        yhat_proc[i] = yhat_proc_inds
        y_proc[i] = gold_proc_inds
    return micro_f1(yhat_proc.ravel(), y_proc.ravel())


def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)

def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc

def recall_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)

def precision_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)

def all_metrics(yhat, y, k=8, yhat_raw=None, calc_auc=True):
    """
        Inputs:
            yhat: binary predictions matrix
            y: binary ground truth matrix
            k: for @k metrics
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "prec", "rec", "f1"]

    #macro
    macro = all_macro(yhat, y)

    #micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + "_macro": macro[i] for i in range(len(macro))}
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})

    #AUC and @k
    if yhat_raw is not None and calc_auc:
        #allow k to be passed as int or list
        if type(k) != list:
            k = [k]
        for k_i in k:
            rec_at_k = recall_at_k(yhat_raw, y, k_i)
            metrics['rec_at_%d' % k_i] = rec_at_k
            prec_at_k = precision_at_k(yhat_raw, y, k_i)
            metrics['prec_at_%d' % k_i] = prec_at_k
            metrics['f1_at_%d' % k_i] = 2*(prec_at_k*rec_at_k)/(prec_at_k+rec_at_k)

        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)

    return metrics
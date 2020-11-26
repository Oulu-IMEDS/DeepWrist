import copy
import math
from pathlib import Path
from sklearn.utils import resample
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, cohen_kappa_score, confusion_matrix, \
    plot_confusion_matrix, precision_recall_curve, average_precision_score
from termcolor import colored
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn.metrics import classification_report
from sklearn import metrics
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.stats import ttest_ind
from classifier.config import get_conf
from classifier.fracture_detector.data import get_train_val_transformations_kneel
from utils import apply_deterministic_computing, apply_fixed_seed, get_snapshots, LandmarkAnnotator, read_dicom, \
    process_xray, create_roi_img, FractureDetector, rotate_image, OODDetector
import pickle

from utils._utils import rotate_point, create_roi_img_with_points, plot_matrix_blue_shades

side_index = {0: 'PA', 1: 'LAT'}
side_to_index = {'PA': 0, 'LAT': 1}
angle_dict = {0: 0, 1: 45, 2: -45}
FONT_SIZE = 9


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)
    # return np.round(values, decs)


def auroc_bootstrap(y, preds, n_bootstrap=5000, seed=12345, model_name='DeepWrist', stratify=True,
                        reproducible=True):
    """Evaluates ROC curve using bootstrapping

    Also reports confidence intervals and prints them.

    Parameters
    ----------
    y : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    savepath: str
        Where to save the figure with ROC curve
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    """

    auc = trunc(roc_auc_score(y, preds), 5)
    np.random.seed(seed)
    aucs = []
    tprs = []
    data = np.concatenate((preds.reshape(-1,1), y.reshape(-1,1)), axis=1)
    base_fpr = np.linspace(0, 1, 1001)

    for i in range(n_bootstrap):
        # ind = np.random.choice(y.shape[0], y.shape[0])
        if stratify:
            st = y
        else:
            st = None
        if reproducible:
            state = seed + i*i
        else:
            state = None
        # resampling seed should not be fixed, otherwise, it will sample exactly the same elements every time
        sample = resample(data, replace=True, random_state=state, n_samples=y.shape[0], stratify=st)
        # if y[ind].sum() == 0:
        #     continue
        if sample[:,1].sum() == 0:
            continue

        aucs.append(roc_auc_score(sample[:,1], sample[:, 0]))

    CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    return auc, CI_l, CI_h


def multi_pr_curve(ys, preds, labels, colors, savepath=None, seed=12345, model_name='DeepWrist', reduction='mean'):

    np.random.seed(seed)

    fig = plt.figure(figsize=(6, 6))
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
    tmp_y = ys[0]
    prevalance = np.sum(tmp_y) / len(tmp_y)
    for y, pred, label, color in zip(ys, preds, labels, colors):
        if reduction == 'mean':
            pred = np.mean(pred, axis=1)
        elif reduction == 'var':
            pred = np.var(pred, axis=1)
        auc = average_precision_score(y, pred)
        precision, recall, _ = precision_recall_curve(y, pred)
        plt.plot(recall, precision, color=color, label=f'{model_name}_{label}, AUPR: {trunc(auc, 2)}')
    plt.plot([0,1], [prevalance, prevalance], '--',color='black', label=f'Prevalence: {trunc(prevalance, 2)}')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(fontsize=FONT_SIZE)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


def multi_roc_curve(ys, preds, labels, colors, savepath=None, seed=12345, model_name='DeepWrist', reduction='mean'):
    """Evaluates ROC curve using bootstrapping

    Also reports confidence intervals and prints them.

    Parameters
    ----------
    y : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    savepath: str
        Where to save the figure with ROC curve
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    """

    np.random.seed(seed)
    fig = plt.figure(figsize=(6, 6))
    for y, pred, label, color in zip(ys, preds, labels, colors):
        if reduction == 'mean':
            pred = np.mean(pred, axis=1)
        elif reduction == 'var':
            pred = np.var(pred, axis=1)
        auc = trunc(roc_auc_score(y, pred), 5)
        fpr, tpr, _ = roc_curve(y, pred, drop_intermediate=False)
        # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
        plt.plot(fpr, tpr, color=color, label=f'{model_name}_{label}' + ', AUROC: %0.2f' % (trunc(auc, 2)))
    plt.plot([0, 1], [0, 1], '--', color='black', label='Random Guess')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(fontsize=FONT_SIZE)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


def roc_curve_bootstrap(y, preds, savepath=None, n_bootstrap=5000, seed=12345, model_name='DeepWrist', stratify=True,
                        reproducible=True):
    """Evaluates ROC curve using bootstrapping

    Also reports confidence intervals and prints them.

    Parameters
    ----------
    y : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    savepath: str
        Where to save the figure with ROC curve
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    """

    auc = trunc(roc_auc_score(y, preds), 5)
    np.random.seed(seed)
    aucs = []
    tprs = []
    data = np.concatenate((preds.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    base_fpr = np.linspace(0, 1, 1001)

    for i in range(n_bootstrap):
        # ind = np.random.choice(y.shape[0], y.shape[0])
        if stratify:
            st = y
        else:
            st = None
        if reproducible:
            state = seed + i * i
        else:
            state = None
        # resampling seed should not be fixed, otherwise, it will sample exactly the same elements every time
        sample = resample(data, replace=True, random_state=state, n_samples=y.shape[0], stratify=st)
        # if y[ind].sum() == 0:
        #     continue
        if sample[:, 1].sum() == 0:
            continue

        aucs.append(roc_auc_score(sample[:, 1], sample[:, 0]))
        # aucs.append(roc_auc_score(y[ind], preds[ind]))
        fpr, tpr, _ = roc_curve(sample[:, 1], sample[:, 0])
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = np.mean(tprs, 0)
    std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    fig = plt.figure(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y, preds, drop_intermediate=False)
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
    plt.plot(fpr, tpr, color='midnightblue', label=f'{model_name}' + ', AUROC: %0.2f' % (trunc(auc, 2)))
    plt.plot([0, 1], [0, 1], '--', color='black', label='Random Guess')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(fontsize=FONT_SIZE)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    return auc, CI_l, CI_h


def pr_curve_bootstrap(y, preds, savepath=None, n_bootstrap=5000, seed=12345, model_name='DeepWrist', stratify=True,
                       reproducible=True):
    precision, recall, _ = precision_recall_curve(y, preds)
    np.random.seed(seed)
    auc = average_precision_score(y, preds)
    aucs = []
    data = np.concatenate((preds.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    for i in range(n_bootstrap):
        if stratify:
            st = y
        else:
            st = None
        if reproducible:
            state = seed + i * i
        else:
            state = None
        sample = resample(data, replace=True, random_state=state, n_samples=y.shape[0], stratify=st)
        # ind = np.random.choice(y.shape[0], y.shape[0])
        # if y[ind].sum() == 0:
        #     continue
        if sample[:, 1].sum() == 0:
            continue
        # aucs.append(average_precision_score(y[ind], preds[ind]))
        aucs.append(average_precision_score(sample[:, 1], sample[:, 0]))

    fig = plt.figure(figsize=(6, 6))
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
    plt.plot(recall, precision, color='midnightblue', label=f'{model_name}, AUPR: {trunc(auc, 2)}')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(fontsize=FONT_SIZE)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    return auc, CI_l, CI_h


def pr_curve_bootstrap_with_readers(y, preds, savepath=None, readers=None, n_bootstrap=5000, seed=12345, inset=True,
                                    stratify=True, reproducible=True):
    precision, recall, _ = precision_recall_curve(y, preds)
    np.random.seed(seed)
    auc = average_precision_score(y, preds)
    aucs = []
    data = np.concatenate((preds.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    for i in range(n_bootstrap):
        if stratify:
            st = y
        else:
            st = None
        if reproducible:
            state = seed + i * i
        else:
            state = None

        sample = resample(data, replace=True, random_state=state, n_samples=y.shape[0], stratify=st)
        # ind = np.random.choice(y.shape[0], y.shape[0])
        # if y[ind].sum() == 0:
        #     continue
        if sample[:, 1].sum() == 0:
            continue
        # aucs.append(average_precision_score(y[ind], preds[ind]))
        aucs.append(average_precision_score(sample[:, 1], sample[:, 0]))
    # fig = plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots(figsize=(6, 6))
    positives = float(np.sum(y))
    total = float(len(y))
    baseline = positives / total
    baseline = trunc(baseline, 2)
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
    ax.plot(recall, precision, color='midnightblue', label=f'DeepWrist, AUPR: {trunc(auc, 2)}')
    ax.plot([0, 1], [baseline, baseline], '--', color='black', label=f'Random Guess (Prevalence: {baseline})')
    if savepath:
        tmp = savepath.split('/')
        tmp2 = tmp[-1]
        tmp3 = tmp2.split('_')
        ts = tmp3[0]
    else:
        ts = 'test'
    with open(f'analysis/{ts}_pr.pkl', 'wb') as f:
        pickle.dump(aucs, f)
        pickle.dump(precision, f)
        pickle.dump(recall, f)

    if readers is not None:
        for key in readers:
            data = readers[key]
            ax.plot([data[1]], [data[0]], marker=data[2],
                    label="%s (%0.2f, %0.2f)" % (key, trunc(data[1], 2), trunc(data[0], 2)), color=data[3])
            # ax_ins.plot([data[1]], [data[0]], marker=data[2], color=data[3])
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid()
    ax.legend(fontsize=FONT_SIZE)
    if inset:
        ax_ins = zoomed_inset_axes(ax, 2.3, loc=8)
        ax_ins.plot(recall, precision, color='midnightblue')
        ax_ins.plot([0, 1], [baseline, baseline], '--', color='black')
    if readers is not None:
        for key in readers:
            data = readers[key]
            # ax.plot([data[1]], [data[0]], marker=data[2], label="%s (%0.2f, %0.2f)"%(key, trunc(data[1], 2), trunc(data[0], 2)), color=data[3])
            if inset:
                ax_ins.plot([data[1]], [data[0]], marker=data[2], color=data[3])
    if inset:
        x1, x2, y1, y2 = 0.8, 1.01, 0.77, 1.01
        ax_ins.set_xlim(x1, x2)
        ax_ins.set_ylim(y1, y2)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
    plt.xticks(visible=True)
    plt.yticks(visible=True)
    if inset:
        mark_inset(ax, ax_ins, loc1=2, loc2=4, fc='none', ec="0.5")
        plt.draw()
    # plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    return auc, CI_l, CI_h


def roc_curve_bootstrap_with_readers(y, preds, savepath=None, n_bootstrap=5000, seed=12345, readers=None, inset=True,
                                     stratify=True, reproducible=True):
    """Evaluates ROC curve using bootstrapping

    Also reports confidence intervals and prints them.

    Parameters
    ----------
    y : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    savepath: str
        Where to save the figure with ROC curve
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed

    """
    auc = trunc(roc_auc_score(y, preds), 5)
    np.random.seed(seed)
    aucs = []
    tprs = []
    data = np.concatenate((preds.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    base_fpr = np.linspace(0, 1, 1001)
    for i in range(n_bootstrap):
        if stratify:
            st = y
        else:
            st = None
        if reproducible:
            state = seed + i * i
        else:
            state = None
        sample = resample(data, replace=True, random_state=state, n_samples=y.shape[0], stratify=st)
        # ind = np.random.choice(y.shape[0], y.shape[0])
        # if y[ind].sum() == 0:
        #     continue
        if sample[:, 1].sum() == 0:
            continue
        # aucs.append(roc_auc_score(y[ind], preds[ind]))
        aucs.append(roc_auc_score(sample[:, 1], sample[:, 0]))
        fpr, tpr, _ = roc_curve(sample[:, 1], sample[:, 0])
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = np.mean(tprs, 0)
    std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    if savepath:
        tmp = savepath.split('/')
        tmp2 = tmp[-1]
        tmp3 = tmp2.split('_')
        ts = tmp3[0]
    else:
        ts = 'test'
    with open(f'analysis/{ts}_roc.pkl', 'wb') as f:
        pickle.dump(aucs, f)
        pickle.dump(tpr, f)
        pickle.dump(fpr, f)
    # fig = plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y, preds, drop_intermediate=False)
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
    ax.plot(fpr, tpr, color='midnightblue', label='DeepWrist, AUROC: %0.2f' % (trunc(auc, 2)))
    ax.plot([0, 1], [0, 1], '--', color='black', label='Random Guess')
    if readers is not None:
        for key in readers:
            data = readers[key]
            ax.plot([1.0 - data[1]], [data[0]], marker=data[2],
                    label=f'{key} ({trunc(data[1], 2)}, {trunc(data[0], 2)})', color=data[3])

    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc=0, bbox_to_anchor=(0.4, 0., 0.4, 0.4), fontsize=FONT_SIZE)
    plt.grid()
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    if inset:
        ax_ins = zoomed_inset_axes(ax, 1.8, loc=10)
        ax_ins.plot(fpr, tpr, color='midnightblue')
        ax_ins.plot([0, 1], [0, 1], '--', color='black')
    if readers is not None:
        for key in readers:
            data = readers[key]
            # ax.plot([data[1]], [data[0]], marker=data[2], label="%s (%0.2f, %0.2f)"%(key, trunc(data[1], 2), trunc(data[0], 2)), color=data[3])
            if inset:
                ax_ins.plot([1.0 - data[1]], [data[0]], marker=data[2], color=data[3])
    if inset:
        x1, x2, y1, y2 = -0.01, 0.46, 0.9, 1.01
        ax_ins.set_xlim(x1, x2)
        ax_ins.set_ylim(y1, y2)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
    plt.xticks(visible=True)
    plt.yticks(visible=True)
    if inset:
        mark_inset(ax, ax_ins, loc1=1, loc2=2, fc='none', ec="0.5")
        plt.draw()
    # plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    return auc, CI_l, CI_h


def accuracy2(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def update_dict(entry, preds, nlls, briers, entropies, sm,  mydict):
    if entry.ID in mydict.keys():
        if entry.Side in mydict[entry.ID].keys():
            mydict[entry.ID][entry.Side]['pred'].append(preds)
            mydict[entry.ID][entry.Side]['nll'].append(nlls)
            mydict[entry.ID][entry.Side]['entropy'].append(entropies)
            mydict[entry.ID][entry.Side]['brier'].append(briers)
            mydict[entry.ID][entry.Side]['sm'].append(sm)
            mydict[entry.ID][entry.Side]['fracture'] = entry.Fracture
            mydict[entry.ID][entry.Side]['ood'] = entry.OOD
        else:
            mydict[entry.ID][entry.Side] = {'pred': [preds], 'nll': [nlls], 'brier': [briers], 'entropy': [entropies],
                                            'fracture': entry.Fracture, 'ood': entry.OOD, 'sm': [sm]}

    else:
        mydict[entry.ID] = dict()
        mydict[entry.ID][entry.Side] = {'pred': [preds], 'nll': [nlls], 'brier': [briers],'entropy': [entropies],
                                        'fracture': entry.Fracture, 'ood': entry.OOD, 'sm': [sm]}


def extract_dict(mydict, side, model_count):
    ids = list()
    preds = list()
    nlls = list()
    briers = list()
    fractures = list()
    oods = list()
    entropies = list()
    sm_outs = list()
    for key in mydict.keys():
        item = mydict[key]
        ids.append(key)
        pred = np.asarray(item[side]['pred'])
        preds.append(np.mean(pred[:, : model_count], axis=0))
        nll = np.asarray(item[side]['nll'])
        nlls.append(np.mean(nll[:, : model_count], axis=0))
        brier = np.asarray(item[side]['brier'])
        briers.append(np.mean(brier[:, : model_count], axis=0))
        fractures.append(item[side]['fracture'])
        oods.append(item[side]['ood'])
        entropy = np.asarray(item[side]['entropy'])
        entropies.append(entropy[:, : model_count])
        sm = np.asarray(item[side]['sm'])
        sm_outs.append(np.mean(sm[:, : model_count], axis=0))

    return np.asarray(ids), np.asarray(preds), np.asarray(nlls), np.asarray(briers), np.asarray(fractures), np.asarray(
        oods), np.asarray(entropies).squeeze(), np.asarray(sm_outs)


def analyse_gt_pred(gt, pred_fracture, print_ok=False, th=0.5):
    pred = (pred_fracture >=th) *1
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i, (g, p) in enumerate(zip(gt, pred)):
        if g == p:
            if g == 1:
                tp += 1
            else:
                tn += 1
        else:
            if g == 1:
                fn +=1
            else:
                fp += 1

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    fnr = fn / (fp + tn)
    fdr = fp / (fn + tn)
    FOR = fn / (fn + tn)
    # pt = (math.sqrt(tpr*(1- tnr)) + tnr -1) / (tpr + tnr -1)
    ts = tp / (tp + fn + fp)
    ba = (tpr + tnr) / 2
    f1 = 2*tp / (2*tp + fp + fn)
    if print_ok:
        print('Sensitivity, Recall, TP Rate: ', tpr)
        print('Specificity, Selectivity, TN Rate:, ', tnr)
        print('Precision, Positive Predictive Value:', ppv)
        print('Negative Predictive Value: ', npv)
        print('Miss Rate, False Negative Rate: ', fnr)
        print('False Discovery Rate: ', fdr)
        print('False Omission Rate: ', FOR)
        # print('Prevalance Threshold: ', pt)
        print('Threat Score, Critical Success Index: ', ts)
        print('Balanced Accuracy ', ba)
        print('F1 Score: ', f1)
    return (tpr, tnr, ppv, f1, ba)


def analyse_annotators(meta, gts, pred_dict):
    data_dict = dict()
    for i in range(meta.shape[0]):
        entry = meta.iloc[i]
        if entry.Side == 0:  # annotation is same for both sides
            for gt in gts:
                if gt in data_dict.keys():
                    data_dict[gt].append(entry[gt])
                else:
                    data_dict[gt] = [entry[gt]]
            pred = np.mean(pred_dict[entry.ID]['pred'])
            pred = (pred >= 0.53) * 1
            if 'DeepWrist' in data_dict.keys():
                data_dict['DeepWrist'].append(pred)
            else:
                data_dict['DeepWrist'] = [pred]

    keys = data_dict.keys()

    accuracy = np.zeros(shape=(len(keys), len(keys)))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            accuracy[i, j] = trunc(accuracy_score(data_dict[k1], data_dict[k2]), 4)


def metric_bootstrap(gt, preds, n_bootstrap=5000, seed=12345, stratify=True, th=0.5):

    (tpr, tnr, ppv, f1, ba) = analyse_gt_pred(gt=gt, pred_fracture=preds, th=th)
    np.random.seed(seed)
    tprs = list()
    tnrs = list()
    ppvs = list()
    f1s = list()
    bas = list()
    data = np.concatenate((gt.reshape(-1, 1), preds.reshape(-1, 1)), axis=1)
    for i in range(n_bootstrap):
        if stratify:
            st = gt
        else:
            st = None
        sample = resample(data, replace=True, n_samples=gt.shape[0], stratify=st)
        # ind = np.random.choice(gt.shape[0], gt.shape[0])
        if sample[:, 0].sum() == 0:
            continue
        (tpr_t, tnr_t, ppv_t, f1_t, ba_t) = analyse_gt_pred(gt=sample[:, 0], pred_fracture=sample[:, 1], print_ok=False)
        tprs.append(tpr_t)
        tnrs.append(tnr_t)
        ppvs.append(ppv_t)
        f1s.append(f1_t)
        bas.append(ba_t)
    CI_L_tpr, CI_H_tpr = np.percentile(tprs, 2.5), np.percentile(tprs, 97.5)
    CI_L_tnr, CI_H_tnr = np.percentile(tnrs, 2.5), np.percentile(tnrs, 97.5)
    CI_L_ppv, CI_H_ppv = np.percentile(ppvs, 2.5), np.percentile(ppvs, 97.5)
    CI_L_f1, CI_H_f1 = np.percentile(f1s,  2.5), np.percentile(f1s, 97.5)
    CI_L_ba, CI_H_ba = np.percentile(bas, 2.5), np.percentile(bas, 97.5)
    out_dict = dict()
    out_dict['tpr'] = [tpr, [CI_L_tpr, CI_H_tpr]]
    out_dict['tnr'] = [tnr, [CI_L_tnr, CI_H_tnr]]
    out_dict['ppv'] = [ppv, [CI_L_ppv, CI_H_ppv]]
    out_dict['f1'] = [f1, [CI_L_f1, CI_H_f1]]
    out_dict['ba'] = [ba, [CI_L_ba, CI_H_ba]]
    return out_dict


def plot_entropy(entropies, ood, save_path):
    id_ind = np.where(ood==0)[0]
    ood_ind = np.where(ood==1)[0]

    for i, o_entropy in enumerate(entropies):
        entropy = copy.copy(o_entropy)
        max_ent = np.max(entropy)
        min_ent = np.min(entropy)
        bins = np.linspace(min_ent, max_ent, 20)
        path = os.path.join(save_path, f'Hist_Ent_{i}.pdf')
        entropy = entropy.squeeze()
        entropy = np.mean(entropy, axis=1)
        id_ent = entropy[id_ind]
        ood_ent = entropy[ood_ind]
        plt.clf()
        plt.hist(id_ent, bins=bins, label='In domain', color='blue', alpha=0.5, edgecolor='k',histtype='bar', rwidth=0.9)
        plt.hist(ood_ent, bins=bins, label='Out of domain', color='green', alpha=0.7, edgecolor='k', histtype='bar', rwidth=0.9)
        plt.xlabel('Entropy')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(path, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # create arguments
    cwd = Path().cwd()
    conf_file = cwd.parents[0] / 'config' / 'config.yaml'
    config = get_conf(conf_file=conf_file, cwd=cwd)
    apply_fixed_seed(seed=config.seed)
    apply_deterministic_computing(config.deterministic)
    meta = pd.read_csv(config.dataset.meta)
    pa = 0
    lat = 1
    if isinstance(config.local_rank, int):
        device = torch.device(f'cuda:{config.local_rank}')
        torch.cuda.set_device(config.local_rank)
    else:
        device = torch.device('cpu')

    with open('temp_old.pkl', 'rb') as f:
        temp_dict = pickle.load(f)

    pa_temp = temp_dict['PA']
    lat_temp = temp_dict['LAT']

    _, val_trf_pa = get_train_val_transformations_kneel(config, meta, side=0)
    _, val_trf_lat = get_train_val_transformations_kneel(config, meta, side=1)
    meta = pd.read_csv(config.dataset.meta)
    oods = meta.OOD.values
    if not os.path.isfile(config.pickle_file):
        # step 1 : process dicom input images

        loc_lat_folder = os.path.join(config.localizer.snapshot_folder, 'LAT')
        loc_pa_folder = os.path.join(config.localizer.snapshot_folder, 'PA')
        fd_lat_folder = os.path.join(config.snapshot_folder, 'LAT')
        fd_pa_folder = os.path.join(config.snapshot_folder, 'PA')

        loc_lat_snapshots = get_snapshots(loc_lat_folder)
        loc_pa_snapshots = get_snapshots(loc_pa_folder)
        fd_lat_snapshots = get_snapshots(fd_lat_folder)
        fd_pa_snapshots = get_snapshots(fd_pa_folder)

        lat_annotator = LandmarkAnnotator(config.localizer, loc_lat_snapshots, side=1, device=device)
        pa_annotator = LandmarkAnnotator(config.localizer, loc_pa_snapshots, side=0, device=device)

        lat_detector = OODDetector(config, fd_lat_snapshots, side=1, device=device, temp=lat_temp, trf=val_trf_lat)
        pa_detector = OODDetector(config, fd_pa_snapshots, side=0, device=device, temp=pa_temp, trf=val_trf_pa)

        # step 2 : localize input images
        data_to_save = list()
        predictions_pa = list()
        gt_pa = list()
        predictions_lat = list()
        gt_lat = list()

        ensemble_dict = dict()
        pbar = tqdm(total=meta.shape[0])

        for i in range(meta.shape[0]):
            pbar.set_description(f'{i + 1} / {meta.shape[0]}')
            pbar.update()
            entry = meta.iloc[i]
            gts_name = list(config.gt)
            gts = [entry[name] for name in gts_name]
            gt = entry[gts_name[0]]
            res = read_dicom(entry.Fname)
            if res is None:
                print(f'Cant read {entry.Fname}')
                continue
            img, spacing, descr, grf = res
            img = np.uint8(process_xray(img))

            img_resized = cv2.resize(img, (256, 256))
            if entry.Side == 0:
                points = pa_annotator.annotate(img_resized)
            else:
                points = lat_annotator.annotate(img_resized)

            img_points = copy.copy(points)
            img_points[:, 0] *= img.shape[1]
            img_points[:, 1] *= img.shape[0]
            a_index = 0
            b_index = 1

            cx = int(np.mean(img_points[:, 0]))
            cy = int(np.mean(img_points[:, 1]))

            # step 3: create roi images
            img_roi, img_roi_points = create_roi_img_with_points(entry, img, spacing, cx, cy, copy.copy(img_points))

            if img_roi is None:
                raise ValueError('roi image is none')
            img_roi = cv2.resize(img_roi, (256, 256))
            img_roi_points *= 256
            # step 4: Detect Fracture
            if entry.Side == 0:
                preds, nlls, briers, entropies, sm = pa_detector.detect(img_roi, entry.Fracture)
                gt_pa.append(entry.Fracture)
            elif entry.Side == 1:
                preds, nlls, briers, entropies, sm = lat_detector.detect(img_roi, entry.Fracture)
                gt_lat.append(entry.Fracture)
            # save gradcam
            yes_no = {0: 'No', 1: 'Yes'}
            update_dict(entry, preds, nlls, briers, entropies, sm, ensemble_dict)
        gt_pa = np.asarray(gt_pa)
        gt_lat = np.asarray(gt_lat)

        with open(config.pickle_file, 'wb') as f:
            pickle.dump(gt_pa, f)
            pickle.dump(gt_lat, f)
            pickle.dump(ensemble_dict, f)
    else:
        # loc_lat_folder = os.path.join(config.localizer.snapshot_folder, 'LAT')
        # loc_pa_folder = os.path.join(config.localizer.snapshot_folder, 'PA')
        # fd_lat_folder = os.path.join(config.snapshot_folder, 'LAT')
        # fd_pa_folder = os.path.join(config.snapshot_folder, 'PA')
        #
        # loc_lat_snapshots = get_snapshots(loc_lat_folder)
        # loc_pa_snapshots = get_snapshots(loc_pa_folder)
        # fd_lat_snapshots = get_snapshots(fd_lat_folder)
        # fd_pa_snapshots = get_snapshots(fd_pa_folder)
        #
        # lat_annotator = LandmarkAnnotator(config.localizer, loc_lat_snapshots, side=1, device=device)
        # pa_annotator = LandmarkAnnotator(config.localizer, loc_pa_snapshots, side=0, device=device)
        #
        # lat_detector = OODDetector(config, fd_lat_snapshots, side=1, device=device)
        # pa_detector = OODDetector(config, fd_pa_snapshots, side=0, device=device)

        # df = pd.read_csv(config.save_path)
        with open(config.pickle_file, 'rb') as f:
            gt_pa = pickle.load(f)
            gt_lat = pickle.load(f)
            ensemble_dict = pickle.load(f)

    print('\nEvaluating Ensemble of PA and LAT models\n')
    ensemble_model_counts = [3,5,7,9]
    pa_probs = list()
    pa_nlls = list()
    pa_briers = list()

    lat_probs = list()
    lat_nlls = list()
    lat_briers = list()

    both_probs = list()
    both_nlls = list()
    both_briers = list()
    both_entropies = list()

    for ens_models in ensemble_model_counts:
        pa_ids, pa_prob, pa_nll, pa_brier, pa_fracture, pa_ood, pa_entropies, pa_sm = extract_dict(ensemble_dict, side=0,
                                                                              model_count=ens_models)
        pa_probs.append(pa_prob)
        pa_nlls.append(pa_nll)
        pa_briers.append(pa_brier)
        lat_ids, lat_prob, lat_nll, lat_brier, lat_fracture, lat_ood, lat_entropies, lat_sm = extract_dict(ensemble_dict, side=1,
                                                                                    model_count=ens_models)
        lat_probs.append(lat_prob)
        lat_nlls.append(lat_nll)
        lat_briers.append(lat_brier)
        both_entropies.append(pa_entropies / 2.0 + lat_entropies / 2.0)
        both_probs.append(lat_prob /2.0 + pa_prob/2.0)
        both_nlls.append(lat_nll/2.0 + pa_nll /2.0 )
        both_briers.append(lat_brier/2.0 + pa_brier/2.0)
        both_ids = lat_ids
        both_fracture = lat_fracture
        both_ood = lat_ood
        last_prob = both_probs[-1]
        last_prob = np.mean(last_prob, axis=1)
        # out_both = metric_bootstrap(both_fracture, last_prob, n_bootstrap=5000, seed=12345, stratify=True, th=0.53)
        # print('Classificaiton BA: ', out_both['ba'])
        # print('Classificaiton Sensitivity: ', out_both['tpr'])
        # last_nll = both_nlls[-1]
        # last_nll = np.mean(last_nll, axis=1)
        # fpr, tpr, th = roc_curve(both_ood, last_nll, drop_intermediate=True)
        # n = len(th)
        #
        # out_both = metric_bootstrap(both_ood, last_nll, n_bootstrap=5000, seed=12345, stratify=True, th=th[n//2])
        # print('OOD NLL BA: ', out_both['ba'])
        # print('OOD Nll Sensitivity: ', out_both['tpr'])
        # th nll [0.42, 0.35, 0.44, 0.38]
        # last_brier = both_briers[-1]
        # last_brier = np.mean(last_brier, axis=1)
        # fpr, tpr, th = roc_curve(both_ood, last_brier, drop_intermediate=True)
        # n = len(th)
        # out_both = metric_bootstrap(both_ood, last_brier, n_bootstrap=5000, seed=12345, stratify=True, th=th[n // 2])
        # print('OOD Brier BA: ', out_both['ba'])
        # print('OOD Brier Sensitivity: ', out_both['tpr'])
        # th brier [0.12, 0.12, 0.15, 0.10]
        auc, lo, hi = auroc_bootstrap(y=both_ood, preds=np.mean(both_entropies[-1], axis=1))
        print(f'Num Model {ens_models} OOD: entropy: AUC: {auc} ({lo} - {hi})')
        last_probs = both_probs[-1]
        predictive_var = np.var(last_probs, axis=1)
        auc, lo, hi = auroc_bootstrap(y=both_ood, preds=predictive_var)
        print(f'Num Model {ens_models}  OOD: Predictive Var: AUC: {auc} ({lo} - {hi})')

    plot_entropy(both_entropies, both_ood,  save_path=config.dataset.ood_home)



    colors = ['green', 'royalblue', 'red', 'midnightblue']


    multi_roc_curve([pa_fracture] * len(ensemble_model_counts), pa_probs, [f'classifier_{i}' for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'PA_Classifier_ROC.pdf'))

    multi_pr_curve([pa_fracture] * len(ensemble_model_counts), pa_probs,
                    [f'classifier_{i}' for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'PA_Classifier_PR.pdf'))
    multi_roc_curve([pa_ood]*len(ensemble_model_counts), pa_nlls, [f'OOD_NLL_{i}' for i in ensemble_model_counts], ['green', 'blue', 'red', 'brown'],
                                                     os.path.join(config.dataset.ood_home, 'PA_OOD_ROC_NLL.pdf'))

    multi_pr_curve([pa_ood] * len(ensemble_model_counts), pa_nlls, [f'OOD_NLL_{i}' for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'PA_OOD_PR_NLL.pdf'))

    multi_roc_curve([pa_ood]*len(ensemble_model_counts), pa_briers, [f'OOD_Brier_{i}' for i in ensemble_model_counts], ['green', 'blue', 'red', 'brown'],
                                                     os.path.join(config.dataset.ood_home, 'PA_OOD_ROC_Brier.pdf'))

    multi_pr_curve([pa_ood] * len(ensemble_model_counts), pa_briers, [f'OOD_Brier_{i}' for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'PA_OOD_PR_Brier.pdf'))


    ## LAT

    multi_roc_curve([lat_fracture] * len(ensemble_model_counts), lat_probs,
                    [f'classifier_{i}' for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'LAT_Classifier_ROC.pdf'))

    multi_pr_curve([lat_fracture] * len(ensemble_model_counts), lat_probs,
                   [f'classifier_{i}' for i in ensemble_model_counts],
                   colors,
                   os.path.join(config.dataset.ood_home, 'LAT_Classifier_PR.pdf'))
    multi_roc_curve([lat_ood] * len(ensemble_model_counts), lat_nlls, [f'OOD_NLL_{i}' for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'LAT_OOD_ROC_NLL.pdf'))

    multi_pr_curve([lat_ood] * len(ensemble_model_counts), lat_nlls, [f'OOD_NLL_{i}' for i in ensemble_model_counts],
                   colors,
                   os.path.join(config.dataset.ood_home, 'LAT_OOD_PR_NLL.pdf'))

    multi_roc_curve([lat_ood] * len(ensemble_model_counts), lat_briers, [f'OOD_Brier_{i}' for i in ensemble_model_counts],
                    ['green', 'blue', 'red', 'brown'],
                    os.path.join(config.dataset.ood_home, 'LAT_OOD_ROC_Brier.pdf'))

    multi_pr_curve([pa_ood] * len(ensemble_model_counts), pa_briers, [f'OOD_Brier_{i}' for i in ensemble_model_counts],
                   colors,
                   os.path.join(config.dataset.ood_home, 'PA_OOD_PR_Brier.pdf'))
    #

    ## both

    multi_roc_curve([both_fracture] * len(ensemble_model_counts), both_probs,
                    [f'classifier_{i}' for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'Both_Classifier_ROC.pdf'), reduction='mean')

    multi_roc_curve([both_ood] * len(ensemble_model_counts), both_probs,
                    [f'OOD_Predictive_Variance_{i}_models'  for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'Both_Pred_Var_ROC.pdf'), reduction='var')

    multi_pr_curve([both_fracture] * len(ensemble_model_counts), both_probs,
                   [f'classifier_{i}' for i in ensemble_model_counts],
                   colors,
                   os.path.join(config.dataset.ood_home, 'Both_Classifier_PR.pdf'), reduction='mean')

    multi_pr_curve([both_ood] * len(ensemble_model_counts), both_probs,
                   [f'OOD_Predictive_Variance_{i}_models' for i in ensemble_model_counts],
                   colors,
                   os.path.join(config.dataset.ood_home, 'Both_Pred_Var_PR.pdf'), reduction='var')

    multi_roc_curve([both_ood] * len(ensemble_model_counts), both_nlls,
                    [f'OOD_NLL_{i}_models' for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'Both_OOD_ROC_NLL.pdf'))

    multi_pr_curve([both_ood] * len(ensemble_model_counts), both_nlls,
                   [f'OOD_NLL_{i}_models' for i in ensemble_model_counts],
                   colors,
                   os.path.join(config.dataset.ood_home, 'Both_OOD_PR_NLL.pdf'))

    multi_roc_curve([both_ood] * len(ensemble_model_counts), both_briers,
                    [f'OOD_Brier_{i}_models' for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'Both_OOD_ROC_Brier.pdf'))

    multi_pr_curve([both_ood] * len(ensemble_model_counts), both_briers,
                   [f'OOD_Brier_{i}_models' for i in ensemble_model_counts],
                   colors,
                   os.path.join(config.dataset.ood_home, 'Both_OOD_PR_Brier.pdf'))

    multi_roc_curve([both_ood] * len(ensemble_model_counts), both_entropies,
                    [f'OOD_Entropy_{i}_models' for i in ensemble_model_counts],
                    colors,
                    os.path.join(config.dataset.ood_home, 'Both_OOD_ROC_Entropy.pdf'))

    multi_pr_curve([both_ood] * len(ensemble_model_counts), both_entropies,
                   [f'OOD_Entropy_{i}_models' for i in ensemble_model_counts],
                   colors,
                   os.path.join(config.dataset.ood_home, 'Both_OOD_PR_Entropy.pdf'))


    # for i in range(len(config.gt)):
    #     # find the threshold:
    #     _, _, th_pa = roc_curve(y_true=gt_pa[:, i], y_score=predictions_pa, drop_intermediate=True)
    #     mid_pa = len(th_pa) // 2
    #     _, _, th_lat = roc_curve(y_true=gt_lat[:, i], y_score=predictions_lat, drop_intermediate=True)
    #     mid_lat = len(th_lat) // 2
    #     _, _, th_ens = roc_curve(y_true=ensemble_gt[:, i], y_score=ensemble_pred, drop_intermediate=True)
    #     mid_ens = len(th_ens) // 2
    #     print(f'\n\n\nEvaluating against {config.gt[i]} as ground truth')
    #     plt.rcParams.update({'font.size': 18})
    #     auc_pa, CI_l_pa, CI_h_pa = roc_curve_bootstrap(gt_pa[:, i], predictions_pa,
    #                                                    os.path.join(config.dataset.test_data_home, 'PA.png'),
    #                                                    n_bootstrap=100)
    #     plt.rcParams.update({'font.size': 18})
    #     auc_lat, CI_l_lat, CI_h_lat = roc_curve_bootstrap(gt_lat[:, i], predictions_lat,
    #                                                       os.path.join(config.dataset.test_data_home, 'LAT.png'),
    #                                                       n_bootstrap=100)
    #
    #     plt.rcParams.update({'font.size': 18})
    #     auc_ens, CI_l_ens, CI_h_ens = roc_curve_bootstrap(ensemble_gt[:, i], ensemble_pred,
    #                                                       os.path.join(config.dataset.test_data_home, 'ensemble.png'),
    #                                                       n_bootstrap=100)
    #
    #
    #     print('PA AUC:', auc_pa)
    #     print(f'PA CI [{CI_l_pa:.5f}, {CI_h_pa:.5f}]')
    #     print('LAT AUC:', auc_lat)
    #     print(f'LAT CI [{CI_l_lat:.5f}, {CI_h_lat:.5f}]')
    #     print('Ens AUC:', auc_ens)
    #     print(f'Ens CI [{CI_l_ens:.5f}, {CI_h_ens:.5f}]')
    #
    #     _, p_val = ttest_ind(predictions_pa, predictions_lat)
    #     displacement = 1
    #     #
    #     # predictions_pa_th = (predictions_pa >= th_pa[mid_pa + displacement])*1  # [1 if x>=0.5 else 0 for x in predictions_pa]
    #     # predictions_lat_th = (predictions_lat >= th_lat[mid_lat + displacement])*1  # [1 if x >= 0.5 else 0 for x in predictions_lat]
    #     # ensemble_pred_th = (ensemble_pred >= th_ens[mid_ens + displacement]) * 1  # [1 if x >= 0.5 else 0 for x in ensemble_pred]
    #     predictions_pa_th = (predictions_pa >= 0.51) * 1
    #     predictions_lat_th = (predictions_lat >= 0.54) * 1
    #     ensemble_pred_th = (ensemble_pred >= 0.525) * 1
    #     print('Accuracy PA: ', accuracy_score(gt_pa[:, i], predictions_pa_th))
    #     print('Accuracy LAT: ', accuracy_score(gt_lat[:, i], predictions_lat_th))
    #     print('Accuracy Ens: ', accuracy_score(ensemble_gt[:, i], ensemble_pred_th))

import copy
import math
from pathlib import Path

import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import  zoomed_inset_axes, mark_inset
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
from utils import apply_deterministic_computing, apply_fixed_seed, get_snapshots, LandmarkAnnotator, read_dicom, \
    process_xray, create_roi_img, FractureDetector, rotate_image
import pickle

from utils._utils import rotate_point, create_roi_img_with_points, plot_matrix_green_shades

side_index = {0: 'PA', 1: 'LAT'}
side_to_index = {'PA': 0, 'LAT': 1}
angle_dict = {0: 0, 1: 45, 2: -45}


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def roc_curve_bootstrap(y, preds, savepath=None, n_bootstrap=5000, seed=12345, model_name='DeepWrist'):
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
    base_fpr = np.linspace(0, 1, 1001)
    for i in range(n_bootstrap):
        ind = np.random.choice(y.shape[0], y.shape[0])
        if y[ind].sum() == 0:
            continue
        aucs.append(roc_auc_score(y[ind], preds[ind]))
        fpr, tpr, _ = roc_curve(y[ind], preds[ind])
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
    plt.plot(fpr, tpr, color='midnightblue',label=f'{model_name}'+', AUROC: %0.2f'%(trunc(auc, 2)))
    plt.plot([0, 1], [0, 1], '--', color='black', label='Random Guess')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(fontsize='small')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    return auc, CI_l, CI_h


def pr_curve_bootstrap(y, preds, savepath=None, n_bootstrap=5000, seed=12345, model_name='DeepWrist'):
    precision, recall, _ = precision_recall_curve(y, preds)
    np.random.seed(seed)
    auc = average_precision_score(y, preds)
    aucs = []
    for i in range(n_bootstrap):
        ind = np.random.choice(y.shape[0], y.shape[0])
        if y[ind].sum() == 0:
            continue
        aucs.append(average_precision_score(y[ind], preds[ind]))

    fig = plt.figure(figsize=(6, 6))
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
    plt.plot(recall, precision, color='midnightblue', label=f'{model_name}, AUPR: {trunc(auc, 2)}')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(fontsize='small')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    return auc, CI_l, CI_h


def pr_curve_bootstrap_with_readers(y, preds, savepath=None, readers=None, n_bootstrap=5000, seed=12345, inset=True):
    precision, recall, _ = precision_recall_curve(y, preds)
    np.random.seed(seed)
    auc = average_precision_score(y, preds)
    aucs = []
    for i in range(n_bootstrap):
        ind = np.random.choice(y.shape[0], y.shape[0])
        if y[ind].sum() == 0:
            continue
        aucs.append(average_precision_score(y[ind], preds[ind]))
    # fig = plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots(figsize=(6,6))
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
            ax.plot([data[1]], [data[0]], marker=data[2], label="%s (%0.2f, %0.2f)"%(key, trunc(data[1], 2), trunc(data[0], 2)), color=data[3])
            # ax_ins.plot([data[1]], [data[0]], marker=data[2], color=data[3])
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid()
    ax.legend(fontsize=8)
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


def roc_curve_bootstrap_with_readers(y, preds, savepath=None, n_bootstrap=5000, seed=12345, readers=None, inset=True):
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
    base_fpr = np.linspace(0, 1, 1001)
    for i in range(n_bootstrap):
        ind = np.random.choice(y.shape[0], y.shape[0])
        if y[ind].sum() == 0:
            continue
        aucs.append(roc_auc_score(y[ind], preds[ind]))
        fpr, tpr, _ = roc_curve(y[ind], preds[ind])
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
    fig, ax = plt.subplots(figsize=(6,6))
    fpr, tpr, _ = roc_curve(y, preds, drop_intermediate=False)
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
    ax.plot(fpr, tpr, color='midnightblue', label='DeepWrist, AUROC: %0.2f'%(trunc(auc, 2)))
    ax.plot([0, 1], [0, 1], '--', color='black', label='Random Guess')
    if readers is not None:
        for key in readers:
            data = readers[key]
            ax.plot([1.0 - data[1]], [data[0]], marker=data[2], label=f'{key} ({trunc(data[1], 2)}, {trunc(data[0], 2)})', color=data[3])

    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc=0, bbox_to_anchor=(0.4, 0., 0.4, 0.4), fontsize=8)
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


def update_dict(entry, pred, mydict, gt):
    if entry.ID in mydict.keys():
        mydict[entry.ID]['pred'].append(pred)
        mydict[entry.ID]['gt'] = gt
    else:
        mydict[entry.ID] = {'pred': [pred], 'gt': gt}


def analyse_gt_pred(gt, pred_fracture, nll=None):
    pred_normal = 1.0 - pred_fracture
    sm = np.stack((pred_normal, pred_fracture), axis=1)
    conf = np.max(sm, axis=1)
    pred = (pred_fracture >=0.5) *1
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    nll_tp = 0
    nll_fp = 0
    nll_tn = 0
    nll_fn = 0
    pred_tp = list()
    pred_fp = list()
    pred_tn = list()
    pred_fn = list()
    for i, (g, p) in enumerate(zip(gt, pred)):
        if g == p:
            if g == 1:
                tp += 1
                pred_tp.append(conf[i])
                if nll is not None:
                    nll_tp += nll[i]
            else:
                tn += 1
                pred_tn.append(conf[i])
                if nll is not None:
                    nll_tn += nll[i]

        else:
            if g == 1:
                fn +=1
                pred_fn.append(conf[i])
                if nll is not None:
                    nll_fn += nll[i]

            else:
                fp += 1
                pred_fp.append(conf[i])
                if nll is not None:
                    nll_fp += nll[i]

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    fnr = fn / (fp + tn)
    fdr = fp / (fn + tn)
    FOR = fn / (fn + tn)
    pt = (math.sqrt(tpr*(1- tnr)) + tnr -1) / (tpr + tnr -1)
    ts = tp / (tp + fn + fp)
    ba = (tpr + tnr) / 2
    f1 = 2*tp / (2*tp + fp + fn)

    print('Sensitivity, Recall, TP Rate: ', tpr)
    print('Specificity, Selectivity, TN Rate:, ', tnr)
    print('Precision, Positive Predictive Value:', ppv)
    print('Negative Predictive Value: ', npv)
    print('Miss Rate, False Negative Rate: ', fnr)
    print('False Discovery Rate: ', fdr)
    print('False Omission Rate: ', FOR)
    print('Prevalance Threshold: ', pt)
    print('Threat Score, Critical Success Index: ', ts)
    print('Balanced Accuracy ', ba)
    print('F1 Score: ', f1)


def analyse_annotators(meta, gts, pred_dict):
    data_dict = dict()
    for i in range(meta.shape[0]):
        entry = meta.iloc[i]
        if entry.Side == 0: # annotation is same for both sides
            for gt in gts:
                if gt in data_dict.keys():
                    data_dict[gt].append(entry[gt])
                else:
                    data_dict[gt] = [entry[gt]]
            pred = np.mean(pred_dict[entry.ID]['pred'])
            pred = (pred >=0.53) * 1
            if 'DeepWrist' in data_dict.keys():
                data_dict['DeepWrist'].append(pred)
            else:
                data_dict['DeepWrist'] = [pred]

    keys = data_dict.keys()
    confusion = np.zeros(shape=(len(keys), len(keys)))
    accuracy = np.zeros(shape=(len(keys), len(keys)))
    pvalue = np.zeros(shape=(len(keys), len(keys)))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            confusion[i, j] = trunc(cohen_kappa_score(data_dict[k1], data_dict[k2], labels=[1, 0], weights='quadratic'), 4)
            accuracy[i, j] = trunc(accuracy_score(data_dict[k1], data_dict[k2]), 4)
            _, p_val = ttest_ind(data_dict[k1], data_dict[k2])
            pvalue[i, j] = trunc(p_val, 6)
    print('Inter Reader Agreement')
    print(confusion)
    labels = ['GT', 'RES', 'R1', 'R2', 'PCP1', 'PCP2', 'PCP3', 'DW']
    plot_matrix_green_shades(confusion, labels, file_to_save='inter_reader_confusion.png')
    print('Inter Reader Accuracy')
    print(accuracy)
    print('Inter Reader P Value')
    print(pvalue)


if __name__ == '__main__':
    # create arguments
    cwd = Path().cwd()
    conf_file = cwd.parents[0] / 'config' / 'config.yaml'
    config = get_conf(conf_file=conf_file, cwd=cwd)
    apply_fixed_seed(seed=config.seed)
    apply_deterministic_computing(config.deterministic)

    pa = 0
    lat = 1
    if not os.path.isfile(config.pickle_file):
        # step 1 : process dicom input images
        meta = pd.read_csv(config.dataset.meta)
        loc_lat_folder = os.path.join(config.localizer.snapshot_folder, 'LAT')
        loc_pa_folder = os.path.join(config.localizer.snapshot_folder, 'PA')
        fd_lat_folder = os.path.join(config.snapshot_folder, 'LAT')
        fd_pa_folder = os.path.join(config.snapshot_folder, 'PA')

        loc_lat_snapshots = get_snapshots(loc_lat_folder)
        loc_pa_snapshots = get_snapshots(loc_pa_folder)
        fd_lat_snapshots = get_snapshots(fd_lat_folder)
        fd_pa_snapshots = get_snapshots(fd_pa_folder)

        lat_annotator = LandmarkAnnotator(config.localizer, loc_lat_snapshots, side=1)
        pa_annotator = LandmarkAnnotator(config.localizer, loc_pa_snapshots, side=0)

        lat_detector = FractureDetector(config, fd_lat_snapshots, side=1)
        pa_detector = FractureDetector(config, fd_pa_snapshots, side=0)

        # step 2 : localize input images
        data_to_save = list()
        predictions_pa = list()
        gt_pa = list()
        predictions_lat = list()
        gt_lat = list()
        sanity_pa_list = list()
        sanity_lat_list = list()
        sanity_tolerance = 0.05
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
                pred, gcam = pa_detector.detect(img_roi)
                predictions_pa.append(pred)
                gt_pa.append(gts)

            elif entry.Side == 1:
                pred, gcam = lat_detector.detect(img_roi)
                predictions_lat.append(pred)
                gt_lat.append(gts)

            # save gradcam
            yes_no = {0: 'No', 1:'Yes'}
            if config.save_gradcam:
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])
                mappable = plt.cm.ScalarMappable(norm=None, cmap=cmap)
                gcam *= 255.0
                gcam_copy = copy.copy(gcam)

                dir = config.gradcam_dir
                os.makedirs(dir, exist_ok=True)

                # plt.figure(figsize=(5, 10))
                # plt.subplot(121)
                fig, (ax1, ax2) = plt.subplots(1,2)
                # plt.title(f'Fracture: {entry.Fracture}')
                # cax = fig.add_axes([0.472, 0.2705, 0.03, 0.4498])
                ax1.imshow(img_roi, cmap=plt.get_cmap('Greys_r'))
                txt = 'Fracture: ' + yes_no[entry.Fracture]
                ax1.text(128,7,s=txt, va='top', ha='center', backgroundcolor='black', color='white')
                ax1.set_xticks([])
                ax1.set_yticks([])
                # cbar1 = plt.colorbar(mappable, cax=cax)
                # cbar1.set_ticks([entry.Fracture])
                # cbar1.set_ticklabels([entry.Fracture])

                # plt.subplot(122)
                # plt.title('Prediction Probability: %0.2f'%trunc(pred, 2))
                ax2.imshow(img_roi, cmap=plt.get_cmap('Greys_r'))
                # cax = fig.add_axes([0.91, 0.2705, 0.03, 0.4498])
                ax2.imshow(gcam, cmap=plt.get_cmap('jet'), alpha=0.3)
                txt = f'Prediction Prob. : {trunc(pred, 2)} '
                ax2.text(128,7,s=txt, va='top', ha='center', backgroundcolor='black', color='white')
                ax2.set_xticks([])
                ax2.set_yticks([])
                # cbar2 = plt.colorbar(mappable, cax=cax)
                # cbar2.set_ticks([trunc(pred, 2)])
                # cbar2.set_ticklabels([trunc(pred, 2)])
                # plt.subplots_adjust(wspace=0.3)
                if config.dataset.name == 'ts_2':
                    fname = os.path.join(dir, f'{entry.Fracture}_{entry.ID}_{entry.Side}.png')
                else:
                    fname = os.path.join(dir, f'{entry.ID}_{entry.Side}.png')
                plt.tight_layout()
                plt.savefig(fname, dpi=300)
                plt.close()
            update_dict(entry, pred, ensemble_dict, gts)
            data_to_save.append([entry.ID, entry.Fname, entry.Side, pred, gts[0]])
        df = pd.DataFrame(data_to_save, columns=['ID', 'Fname', 'Side', 'Fracture_Prediction', 'Fracture'] )

        df.to_csv(config.save_path, index=None)

        gt_pa = np.asarray(gt_pa)
        gt_lat = np.asarray(gt_lat)

        predictions_pa = np.asarray(predictions_pa)
        predictions_lat = np.asarray(predictions_lat)
        sanity_pa = np.asarray(sanity_pa_list)
        sanity_lat = np.asarray(sanity_lat_list)

        with open(config.pickle_file, 'wb') as f:
            pickle.dump(gt_pa, f)
            pickle.dump(gt_lat, f)
            pickle.dump(predictions_pa, f)
            pickle.dump(predictions_lat, f)
            pickle.dump(ensemble_dict, f)
    else:
        meta = pd.read_csv(config.dataset.meta)
        loc_lat_folder = os.path.join(config.localizer.snapshot_folder, 'LAT')
        loc_pa_folder = os.path.join(config.localizer.snapshot_folder, 'PA')
        fd_lat_folder = os.path.join(config.snapshot_folder, 'LAT')
        fd_pa_folder = os.path.join(config.snapshot_folder, 'PA')

        loc_lat_snapshots = get_snapshots(loc_lat_folder)
        loc_pa_snapshots = get_snapshots(loc_pa_folder)
        fd_lat_snapshots = get_snapshots(fd_lat_folder)
        fd_pa_snapshots = get_snapshots(fd_pa_folder)

        lat_annotator = LandmarkAnnotator(config.localizer, loc_lat_snapshots, side=1)
        pa_annotator = LandmarkAnnotator(config.localizer, loc_pa_snapshots, side=0)

        lat_detector = FractureDetector(config, fd_lat_snapshots, side=1)
        pa_detector = FractureDetector(config, fd_pa_snapshots, side=0)

        df = pd.read_csv(config.save_path)
        with open(config.pickle_file, 'rb') as f:
            gt_pa = pickle.load(f)
            gt_lat = pickle.load(f)
            predictions_pa = pickle.load(f)
            predictions_lat = pickle.load(f)
            ensemble_dict = pickle.load(f)

    print('\nEvaluating Ensemble of PA and LAT models\n')
    ensemble_pred = list()
    ensemble_gt = list()
    for key in ensemble_dict:
        value = ensemble_dict[key]
        preds = value['pred']
        gt = value['gt']
        pred = np.mean(preds)
        ensemble_pred.append(pred)
        ensemble_gt.append(gt)
    ensemble_gt = np.asarray(ensemble_gt)
    ensemble_pred = np.asarray(ensemble_pred)


    nll_pa = nll_uncertainty(predictions_pa)
    nll_lat = nll_uncertainty(predictions_lat)
    nll_ens = nll_uncertainty(ensemble_pred)

    for i in range(len(config.gt)):
        # find the threshold:
        _, _, th_pa = roc_curve(y_true=gt_pa[:, i], y_score=predictions_pa, drop_intermediate=True)
        mid_pa = len(th_pa) // 2
        _, _, th_lat = roc_curve(y_true=gt_lat[:, i], y_score=predictions_lat, drop_intermediate=True)
        mid_lat = len(th_lat) // 2
        _, _, th_ens = roc_curve(y_true=ensemble_gt[:, i], y_score=ensemble_pred, drop_intermediate=True)
        mid_ens = len(th_ens) // 2
        print(f'\n\n\nEvaluating against {config.gt[i]} as ground truth')
        plt.rcParams.update({'font.size': 18})
        auc_pa, CI_l_pa, CI_h_pa = roc_curve_bootstrap(gt_pa[:, i], predictions_pa,
                                                       os.path.join(config.dataset.test_data_home, 'PA.png'),
                                                       n_bootstrap=100)
        plt.rcParams.update({'font.size': 18})
        auc_lat, CI_l_lat, CI_h_lat = roc_curve_bootstrap(gt_lat[:, i], predictions_lat,
                                                          os.path.join(config.dataset.test_data_home, 'LAT.png'),
                                                          n_bootstrap=100)

        plt.rcParams.update({'font.size': 18})
        auc_ens, CI_l_ens, CI_h_ens = roc_curve_bootstrap(ensemble_gt[:, i], ensemble_pred,
                                                          os.path.join(config.dataset.test_data_home, 'ensemble.png'),
                                                          n_bootstrap=100)


        print('PA AUC:', auc_pa)
        print(f'PA CI [{CI_l_pa:.5f}, {CI_h_pa:.5f}]')
        print('LAT AUC:', auc_lat)
        print(f'LAT CI [{CI_l_lat:.5f}, {CI_h_lat:.5f}]')
        print('Ens AUC:', auc_ens)
        print(f'Ens CI [{CI_l_ens:.5f}, {CI_h_ens:.5f}]')

        _, p_val = ttest_ind(predictions_pa, predictions_lat)
        displacement = 1
        #
        # predictions_pa_th = (predictions_pa >= th_pa[mid_pa + displacement])*1  # [1 if x>=0.5 else 0 for x in predictions_pa]
        # predictions_lat_th = (predictions_lat >= th_lat[mid_lat + displacement])*1  # [1 if x >= 0.5 else 0 for x in predictions_lat]
        # ensemble_pred_th = (ensemble_pred >= th_ens[mid_ens + displacement]) * 1  # [1 if x >= 0.5 else 0 for x in ensemble_pred]
        predictions_pa_th = (predictions_pa >= 0.51) * 1
        predictions_lat_th = (predictions_lat >= 0.54) * 1
        ensemble_pred_th = (ensemble_pred >= 0.525) * 1
        print('Accuracy PA: ', accuracy_score(gt_pa[:, i], predictions_pa_th))
        print('Accuracy LAT: ', accuracy_score(gt_lat[:, i], predictions_lat_th))
        print('Accuracy Ens: ', accuracy_score(ensemble_gt[:, i], ensemble_pred_th))

    # if len(config.gt) >1:
    #     # confusion matrix
    #     # confusion = confusion_matrix(y_true=gt_pa[:, 0], y_pred=gt_pa[:, 1]) # radiologist vs resident
    #     # print(f'\n\nConfusion Matrix with Radiologist as GT, resident as prediciton')
    #     # print(confusion)
    confusion = confusion_matrix(y_true=gt_pa[:, 0], y_pred=predictions_pa_th, labels=[1,0])  # radiologist vs AI
    print(f'Confusion Matrix with Radiologist as GT, AI as prediciton for PA')
    print(confusion)
    confusion = confusion_matrix(y_true=gt_lat[:, 0], y_pred=predictions_lat_th, labels=[1,0])  # radiologist vs AI
    print(f'Confusion Matrix with Radiologist as GT, AI as prediciton for LAT')
    print(confusion)

    confusion = confusion_matrix(y_true=ensemble_gt[:, 0], y_pred=ensemble_pred_th, labels=[1,0])  # radiologist vs AI
    print(f'Confusion Matrix for both view with Radiologist as GT, AI as prediciton')
    print(confusion)

    print('Analysing Readers')
    analyse_annotators(meta, list(config.gt), ensemble_dict)

    # save the false positive and true netagive
    if config.save_defect:
        detects_to_save = list()
        save_dir = config.defect_dir
        faulty_pa = os.path.join(save_dir, 'pa')
        faulty_lat = os.path.join(save_dir, 'lat')
        os.makedirs(faulty_lat, exist_ok=True)
        os.makedirs(faulty_pa, exist_ok=True)
        for i in range(df.shape[0]):
            entry = df.iloc[i]
            gt = entry.Fracture

            if entry.Side == 0:
                # pred = (entry.Fracture_Prediction > th_pa[mid_pa+ displacement]) * 1
                pred = (entry.Fracture_Prediction > 0.5) * 1
            else:
                # pred = (entry.Fracture_Prediction > th_lat[mid_lat + displacement]) * 1
                pred = (entry.Fracture_Prediction > 0.5) * 1
            if gt == 1 and pred != gt: # false netative
                detects_to_save.append([entry.ID, entry.Fname, entry.Side, entry.Fracture])
                if entry.Side == 0:
                    fn_folder = os.path.join(faulty_pa, 'fn')
                else:
                    fn_folder = os.path.join(faulty_lat, 'fn')
                os.makedirs(fn_folder, exist_ok=True)
                tmp = entry.Fname.split('/')
                filename = os.path.join(fn_folder, f'{tmp[-4]}_{i}_roi.png' )
                img_name = os.path.join(fn_folder, f'{tmp[-4]}_{i}.png')
                res = read_dicom(entry.Fname)
                if res is None:
                    print(f'Cant read {entry.Fname}')
                    continue
                img, spacing, descr, grf = res
                img = np.uint8(process_xray(img))
                # if entry.Angle != 0:
                #     rot = -1 * angle_dict[entry.Angle]
                #     img = rotate_image(img, rot)
                img_resized = cv2.resize(img, (256, 256))

                if entry.Side == 0:
                    points = pa_annotator.annotate(img_resized)
                else:
                    points = lat_annotator.annotate(img_resized)
                points[:, 0] *= img.shape[1]
                points[:, 1] *= img.shape[0]
                a_index = 0

                b_index = 1

                cx = int(np.mean(points[:, 0]))
                cy = int(np.mean(points[:, 1]))

                if points[b_index, 0] > points[a_index, 0]:
                    dx = points[b_index, 1] - points[a_index, 1]
                    dy = points[b_index, 0] - points[a_index, 0]
                else:
                    dx = points[a_index, 1] - points[b_index, 1]
                    dy = points[a_index, 0] - points[b_index, 0]
                rad_angle = np.arctan2(dx, dy)
                deg_angle = rad_angle / np.pi * 180.0

                # step 3: create roi images
                img_roi = create_roi_img(entry, img, spacing, cx, cy)
                if img_roi is None:
                    raise ValueError('roi image is none')
                img_roi = cv2.resize(img_roi, (256, 256))
                # cv2.imwrite(img_name, img)
                # plt.clf()
                # plt.imshow(img, cmap='gray')
                # plt.scatter([cx], [cy],  marker='o', color="blue")
                # plt.savefig(img_name, dpi=300)
                cv2.imwrite(filename, img_roi)
            elif gt == 0 and pred != gt:
                detects_to_save.append([entry.ID, entry.Fname, entry.Side, entry.Fracture])
                if entry.Side == 0:
                    fp_folder = os.path.join(faulty_pa, 'fp')
                else:
                    fp_folder = os.path.join(faulty_lat, 'fp')
                os.makedirs(fp_folder, exist_ok=True)
                tmp = entry.Fname.split('/')
                filename = os.path.join(fp_folder, f'{tmp[-4]}_{i}_roi.png')
                img_name = os.path.join(fp_folder, f'{tmp[-4]}_{i}.png')
                res = read_dicom(entry.Fname)
                if res is None:
                    print(f'Cant read {entry.Fname}')
                    continue
                img, spacing, descr, grf = res
                img = np.uint8(process_xray(img))
                # if entry.Angle != 0:
                #     rot = -1 * angle_dict[entry.Angle]
                #     img = rotate_image(img, rot)
                img_resized = cv2.resize(img, (256, 256))
                if entry.Side == 0:
                    points = pa_annotator.annotate(img_resized)
                else:
                    points = lat_annotator.annotate(img_resized)
                points[:, 0] *= img.shape[1]
                points[:, 1] *= img.shape[0]
                a_index = 0

                b_index = 1

                cx = int(np.mean(points[:, 0]))
                cy = int(np.mean(points[:, 1]))

                if points[b_index, 0] > points[a_index, 0]:
                    dx = points[b_index, 1] - points[a_index, 1]
                    dy = points[b_index, 0] - points[a_index, 0]
                else:
                    dx = points[a_index, 1] - points[b_index, 1]
                    dy = points[a_index, 0] - points[b_index, 0]
                rad_angle = np.arctan2(dx, dy)
                deg_angle = rad_angle / np.pi * 180.0
                # if np.abs(deg_angle) >= 30:
                #     # rotate the points and images
                #     ic = (img.shape[1] // 2, img.shape[0] // 2)
                #     cy, cx = rotate_point((cy, cx), ic, rad_angle)
                #     img = rotate_image(img,  deg_angle)
                # step 3: create roi images
                img_roi = create_roi_img(entry, img, spacing, cx, cy)
                if img_roi is None:
                    raise ValueError('roi image is none')
                img_roi = cv2.resize(img_roi, (256, 256))

                # cv2.imwrite(img_name, img)
                # plt.clf()
                # plt.imshow(img, cmap='gray')
                # plt.scatter([cx], [cy],  marker='o', color="blue")
                # plt.savefig(img_name, dpi=300)
                cv2.imwrite(filename, img_roi)
        # detects_to_save = pd.DataFrame(detects_to_save, columns=['ID', 'Fname', 'Side', 'Fracture'])
        # detects_to_save.to_csv('/home/backbencher/DATA/wrist_data/preprocessed_fixed/defects.csv', index=None)



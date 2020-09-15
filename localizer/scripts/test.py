import copy
import pickle
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd

from classifier.scripts.test import trunc
from localizer.config import get_conf
from utils import get_snapshots, LandmarkAnnotator, read_dicom, process_xray, apply_fixed_seed, \
    apply_deterministic_computing
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import metrics


def generate_pr_curve(distances, seed=12345, n_bootstrap=5000, savepath=None):
    distances = np.asarray(distances).reshape(-1)
    prec = list()
    rec = list()
    N = len(distances)
    for precision in range(1, 101, 1):
        precision /= 10.0
        d = np.sum(1.0 * (distances <= precision)) / N
        prec.append(precision)
        rec.append(d)
    auc = metrics.auc(rec, prec)
    # precision, recall, _ = precision_recall_curve(y, preds)
    np.random.seed(seed)
    # auc = average_precision_score(y, preds)
    precision_list = [1,2,3,4,5]
    for p in precision_list:
        aucs = []
        recalls = list()
        recall = np.sum(1.0 * (distances <= p)) / N
        y = 1 * (distances <= p)
        for i in range(n_bootstrap):
            ind = np.random.choice(distances.shape[0], distances.shape[0])
            if y[ind].sum() == 0:
                continue
            r = y[ind].sum() / len(ind)
            recalls.append(r)
        CI_l, CI_h = np.percentile(recalls, 2.5), np.percentile(recalls, 97.5)
        print(f'Recall : {recall} ({CI_l} - {CI_h}) at Precision {p}mm ')
    fig = plt.figure(figsize=(6, 6))
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
    plt.plot(rec[::-1], prec[::-1],  color='midnightblue', label=f'Landmark Localizer')

    plt.xlim([0, 1])
    plt.ylim([10,1])
    plt.grid()
    plt.legend(fontsize='small')
    plt.xlabel('Recall (correctly classified points)')
    plt.ylabel('Precision (millimeters)')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)
    # CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    return auc

if __name__ == '__main__':

    cwd = Path().cwd()
    conf_file = cwd.parents[0] / 'config' / 'config.yaml'
    config = get_conf(conf_file=conf_file, cwd=cwd)

    apply_fixed_seed(config.seed)
    apply_deterministic_computing(config.deterministic)
    lat_folder = os.path.join(config.snapshot_folder, 'LAT')
    pa_folder = os.path.join(config.snapshot_folder, 'PA')

    lat_snapshots = get_snapshots(lat_folder)
    pa_snapshots = get_snapshots(pa_folder)
    metadata = pd.read_csv(config.dataset.meta)

    lat_annotator = LandmarkAnnotator(config, lat_snapshots, side=1)
    pa_annotator = LandmarkAnnotator(config, pa_snapshots, side=0)
    annotated_meta = list()
    if config.save_image:
        os.makedirs(config.save_image_dir, exist_ok=True)

    pbar = tqdm(metadata.shape[0])
    mm2 = 0
    mm3 = 0
    mm4 = 0
    mm5 = 0
    correct = 0
    dist_list = list()
    if os.path.exists('dist.pkl'):
        with open('dist.pkl', 'rb') as f:
            dist_list = pickle.load(f)
    else:
        for i in range(metadata.shape[0]):
            pbar.set_description(f'{i + 1} / {metadata.shape[0]}')
            pbar.update()
            entry = metadata.iloc[i]
            img = cv2.imread(entry.Fname, 0)
            dicom_file = entry.DICOM
            data, pixel_spacing, _, _ = read_dicom(dicom_file)
            original_img = process_xray(data)

            img_resized = cv2.resize(img, (256,256))
            if entry.Side == 0:
                landmarks = pa_annotator.annotate(img_resized)
            else:
                landmarks = lat_annotator.annotate(img_resized)

            gt = eval(entry.Points)
            gt = np.asarray(gt, dtype=np.float32)
            gt[:, 0] /= img.shape[1]
            gt[:, 1] /= img.shape[0]

            gt[:, 0] *= original_img.shape[1]
            gt[:, 1] *= original_img.shape[0]
            points = copy.copy(landmarks)
            points[:, 0] = (points[:,0]*original_img.shape[1])
            points[:, 1] =(points[:, 1]*original_img.shape[0])
            diff = points - gt
            diff *= pixel_spacing
            dist = np.sqrt(np.sum(diff**2, axis=1))
            dist = np.mean(dist)
            dist_list.append(dist)
            if config.save_image:
                plt.figure(figsize=(10,10))
                plt.imshow(img, cmap=plt.get_cmap('Greys_r'))
                plt.scatter(gt[:, 0], gt[:, 1], marker='H', color='blue', s=200)
                plt.scatter(points[:, 0], points[:, 1], marker='o', color='green', s=200)
                file = os.path.join(config.save_image_dir, '%03d.png'%i)
                plt.savefig(file, dpi=300)
                plt.clf()
                plt.close()
        with open(f'dist.pkl', 'wb') as f:
            pickle.dump(dist_list, f)

    generate_pr_curve(dist_list, savepath='suppli_lm_pr.pdf')

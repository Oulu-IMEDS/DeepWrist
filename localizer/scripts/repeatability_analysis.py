import os
import pandas as pd
import numpy as np
from sklearn import metrics

from utils import read_dicom, process_xray


def generate_ci(distances, n_bootstrap=5000):
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
    # CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    return auc


if __name__ == '__main__':
    original_file = '/home/backbencher/DATA/wrist_data/landmark_data/landmark_meta_dicom.csv'
    lm100 = '/home/backbencher/DATA/wrist_data/landmark_data/lm100.csv'
    orig_meta = pd.read_csv(original_file)
    lm100_meta = pd.read_csv(lm100)
    lm100_dict = lm100_meta.to_dict()
    filenames = list(lm100_dict['Fname'].values())
    orig_dict = dict()
    for i in range(orig_meta.shape[0]):
        entry = orig_meta.iloc[i]
        if entry.Fname in filenames:
            orig_dict[entry.Fname] = [entry.Points, entry.DICOM]

    mm1 = 0
    mm2 = 0
    mm3 = 0
    mm4 = 0
    mm5 = 0

    mmx1 = 0
    mmx2 = 0
    mmx3 = 0
    mmx4 = 0
    mmx5 = 0

    mmy1 = 0
    mmy2 = 0
    mmy3 = 0
    mmy4 = 0
    mmy5 = 0
    correct = 0
    dist_list = list()
    distx_list = list()
    disty_list = list()
    for i in range(lm100_meta.shape[0]):
        if entry.Side < 2:
            entry = lm100_meta.iloc[i]
            dicom = orig_dict[entry.Fname][1]
            data, pixel_spacing, _, _ = read_dicom(dicom)
            original_img = process_xray(data)
            shape = original_img.shape
            pointsA = np.asarray(eval(entry.Points), dtype=np.float32) / 256
            pointsB = np.asarray(eval(orig_dict[entry.Fname][0]), dtype=np.float32) / 256
            pointsA[:, 0] *= shape[1]
            pointsA[:, 1] *= shape[0]

            pointsB[:, 0] *= shape[1]
            pointsB[:, 1] *= shape[0]
            diffX = pointsA[:, 0].reshape(3,1) - pointsB[:, 0].reshape(3,1)
            diffY = pointsA[:, 1].reshape(3, 1) - pointsB[:, 1].reshape(3, 1)
            diff = pointsA - pointsB
            diff *= pixel_spacing
            diffX *= pixel_spacing
            diffY *= pixel_spacing
            dist = np.sqrt(np.sum(diff**2, axis=1))
            dist = np.mean(dist)
            distX = np.sqrt(np.sum(diffX ** 2, axis=1))
            distX = np.mean(distX)
            distY = np.sqrt(np.sum(diffY ** 2, axis=1))
            distY = np.mean(distY)
            dist_list.append(dist)
            distx_list.append(distX)
            disty_list.append(distY)

    cases = lm100_meta.shape[0]
    print('For both axis: ')
    generate_ci(np.asarray(dist_list).reshape(-1))
    print('\nFor X axis:')
    generate_ci(np.asarray(distx_list).reshape(-1))
    print('\nFor Y axis: ')
    generate_ci(np.asarray(disty_list).reshape(-1))





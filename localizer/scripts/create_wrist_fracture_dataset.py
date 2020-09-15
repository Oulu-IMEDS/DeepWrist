import copy
import math

from localizer.config import get_conf
from pathlib import Path
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import utils as dwutils
from utils import LandmarkAnnotator
import cv2
import matplotlib.pylab as plt

from utils._utils import create_roi_img_with_points

IND_TO_SIDE = {0: 'PA', 1: 'LAT'}


def get_snapshots(folder):
    snapshot_list = list()
    for root, dirs, files in os.walk(folder):
        for file in files:
            if '.ckpt' in file:
                snapshot_list.append(os.path.join(root, file))
    return snapshot_list



if __name__ == '__main__':
    cwd = Path().cwd()
    conf_file = cwd.parents[0] / 'config' / 'config.yaml'
    config = get_conf(conf_file=conf_file, cwd=cwd)
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
    save_dir = config.dataset.save_path
    pa_dir = os.path.join(save_dir, 'preprocessed_fixed','PA')
    lat_dir = os.path.join(save_dir, 'preprocessed_fixed', 'LAT')
    os.makedirs(pa_dir, exist_ok=True)
    os.makedirs(lat_dir, exist_ok=True)
    pbar = tqdm(total=metadata.shape[0])
    for i in range(metadata.shape[0]):
        entry = metadata.iloc[i]
        pbar.set_description(f'{i} / {metadata.shape[0]}')
        pbar.update()
        res = dwutils.read_dicom(entry.Fname)
        if res is None:
            print(f'Cant read {entry.Fname}')
            continue
        img, spacing, descr, grf = res
        img = np.uint8(dwutils.process_xray(img))
        img_resized = cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)
        if entry.Side == 0:
            points = pa_annotator.annotate(img_resized)
        else:
            points = lat_annotator.annotate(img_resized)

        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]

        pad = 400
        roi_size = 90
        pad_top = 20
        if entry.Side == 0:
            roi_size = 70  # For lateral images we do not need a very big ROI
            pad_top = 15

        h, w = img.shape
        tmp = np.zeros((h + pad, w + pad), dtype=np.uint8)
        tmp[pad // 2:-pad // 2, pad // 2:-pad // 2] = img
        img_orig_padded = tmp
        cx = int(np.mean(points[:, 0]))
        cy = int(np.mean(points[:, 1]))


        # step 3: create roi images
        img_roi, img_roi_points = create_roi_img_with_points(entry, img, spacing, cx, cy, copy.copy(points))

        if img_roi is None:
            print('Img roi is none')
        img_roi = cv2.resize(img_roi, (256, 256))

        if entry.Side == 0: # PA
            img_file = os.path.join(pa_dir, f'{entry.ID}_{IND_TO_SIDE[entry.Side]}.png')
        elif entry.Side == 1:
            img_file = os.path.join(lat_dir, f'{entry.ID}_{IND_TO_SIDE[entry.Side]}.png')

        cv2.imwrite(img_file, img_roi)
    pbar.close()

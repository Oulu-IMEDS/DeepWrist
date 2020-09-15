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
    for i in range(metadata.shape[0]):
        entry = metadata.iloc[i]
        res = dwutils.read_dicom(entry.Fname)
        if res is None:
            print(f'Cant read {entry.Fname}')
            continue
        img, spacing, descr, grf = res
        img = np.uint8(dwutils.process_xray(img))
        img_resized = cv2.resize(img, (256,256))
        if entry.Side == 0:
            cx, cy = pa_annotator.annotate(img_resized, img.shape[0], img.shape[1])
        else:
            cx, cy = lat_annotator.annotate(img_resized, img.shape[0], img.shape[1])
        cx = int(cx*img.shape[1])
        cy = int(cy*img.shape[0])
        if config.save_image:
            plt.figure(figsize=(10,10))
            plt.imshow(img, cmap=plt.get_cmap('Greys_r'))
            plt.scatter(cx, cy, marker='H', color='blue', s=200)
            file = os.path.join(config.save_image_dir, '%03d.png'%i)
            plt.savefig(file, dpi=300)
            plt.clf()
            plt.close()
            annotated_meta.append([entry.Fname, entry.Side,  cx, cy])

    meta = pd.DataFrame(annotated_meta, columns=['Fname', 'Side', 'cx', 'cy'])
    meta.to_csv(config.save_path, index=None)

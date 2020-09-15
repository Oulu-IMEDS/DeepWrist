import copy
import os
import pandas as pd
import cv2
import pydicom as dicom
import numpy as np
import solt.core as slc
import solt.data as sld
import solt.transforms as slt
IND_TO_SIDE = {0: 'PA', 1: 'LAT'}


def get_meta(config):
    meta_path = os.path.join(str(config.dataset.data_home), str(config.dataset.meta))
    meta = pd.read_csv(meta_path)

    def sanity_check(entry):
        side_txt = IND_TO_SIDE[entry.Side]
        filename = os.path.join(config.dataset.data_home, IND_TO_SIDE[entry.Side], f'{entry.ID}_{side_txt}.png')
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return img is not None

    print(f'Before Sanity Check: {meta.shape[0]} entries')
    meta = meta[meta.apply(sanity_check, axis=1)]
    print(f'After Sanity Check: {meta.shape[0]} entrise ')

    return meta

def read_dicom(filename):
    # This function tries to read the dicom file

    try:
        data = dicom.read_file(filename)
        img = np.frombuffer(data.PixelData, dtype=np.uint16).astype(float)

        if data.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max() - img
        img = img.reshape((data.Rows, data.Columns))
    except:
        return None
    try:
        graphics = data[0x2001, 0x9000][0][0x0070, 0x0001][0][0x0070, 0x0008][0][0x0070, 0x0006].value
    except:
        graphics = None

    descr = data.StudyDescription
    try:
        return img, float(data.ImagerPixelSpacing[0]), descr, graphics
    except:
        pass
    try:
        return img, float(data.PixelSpacing[0]), descr, graphics
    except:
        return None

def wrap_img_target_solt(img, target):
    if not isinstance(img, np.ndarray):
        raise TypeError

    if len(img.shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    if len(img.shape) != 3:
        raise ValueError

    return sld.DataContainer((img, target), fmt='IL')


def solt_to_img_target(dc: sld.DataContainer):
    if dc.data_format != 'IL':
        raise TypeError
    return dc.data


def apply_by_index(data, transform, idx=0):
    """Applies callable to certain objects in iterable using given indices.

    Parameters
    ----------
    data: tuple or list
    transform: callable
    idx: int or tuple or or list None

    Returns
    -------
    result: tuple

    """
    if isinstance(data, sld.DataContainer):
        data = data.data
    if idx is None:
        return data
    if not isinstance(data, (tuple, list)):
        raise TypeError
    if not isinstance(idx, (int, tuple, list)):
        raise TypeError

    if isinstance(idx, int):
        idx = [idx, ]

    idx = set(idx)
    res = []
    for i, item in enumerate(data):
        if i in idx:
            res.append(transform(item))
        else:
            res.append(copy.deepcopy(item))

    return tuple(res)


def center_crop(img, size):
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape

    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    img = img[y1:y1 + size[1], x1:x1 + size[0]]
    return img


def five_crop(img, size):
    img = img.copy()
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, c = img.shape
    # get central crop
    c_cr = center_crop(img, (size, size))
    # upper-left crop
    ul_cr = img[0:size, 0:size]
    # upper-right crop
    ur_cr = img[0:size, w - size:w]
    # bottom-left crop
    bl_cr = img[h - size:h, 0:size]
    # bottom-right crop
    br_cr = img[h - size:h, w - size:w]
    return np.stack((c_cr, ul_cr, ur_cr, bl_cr, br_cr))

if __name__ == '__main__':
    fname = '/home/backbencher/DATA/wrist_data/Rais/train/detection_raw_data/fractures_anonymized/PAT00001/STU00000/SER00000/IMG00000'
    res = read_dicom(fname)
    img, spacing, descr, grf = res
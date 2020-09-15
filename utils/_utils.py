"""
General tools, needed for data processing when you deal with DICOM files.

"""
import copy
import math
import os
import random
import seaborn as sn
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import solt.data as sld
import torch
from matplotlib import colors

# do not remove
from classifier.fracture_detector.model import SeResNet_PTL
from localizer.kneel_before_wrist.model import HourglassNet_PTL


def read_dicom(filename):
    """
    Parameters
    ----------
    filename: str
        path to the dicom file
    Returns
    -------
        multiple return values containing the 16 bit image, pixel spacing, dicom description and dicom graphics info.

    """
    try:
        data = dicom.read_file(filename)
        img = np.frombuffer(data.PixelData, dtype=np.uint16).astype(float)

        if data.PhotometricInterpretation == 'MONOCHROME1':
            img = 4095 - img
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


def process_xray(img, cut_min=5, cut_max=99, multiplier=255):
    """
    This function changes the histogram of the image by doing global contrast normalization
    Parameters
    ----------
    img: numpy.ndarray
        16 bit image
    cut_min: int
        lowest percentile to cut the img histogram
    cut_max: int
        highest percentile to cut the img histogram
    multiplier: int
        highest value in the image scale, for 8 bit image it is 255, for 16 bit it is 4095, for floating point it is 1.0

    Returns
    -------

    """

    img = img.copy()
    lim1, lim2 = np.percentile(img, [cut_min, cut_max])
    img[img < lim1] = lim1
    img[img > lim2] = lim2

    img -= lim1
    img /= img.max()
    img *= multiplier

    return img


def img_crop(img, x1: int, y1: int, size: tuple):
    """

    Parameters
    ----------
    img: numpy.ndrarray
        image to crop
    x1: int
        position on X axis to start crop
    y1: int
        position on Y axis to start crop
    size: tuple of int
        size of cropped image

    Returns
    -------
        numpy.ndarray
            cropped image.
    """
    h, w, c = img.shape
    pad_w = 0
    pad_h = 0
    if w < size[0] and h < size[1]:
        pad_w = (size[0] - w) // 2
    if h < size[1]:
        pad_h = (size[1] - h) // 2
    img_pad = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    if len(img_pad.shape) == 2:
        img_pad = np.expand_dims(img_pad, -1)

    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad


def npg2tens(x, type='f'):
    """
    Numpy grayscale to float tensor
    """
    if type == 'f':
        return torch.from_numpy(x).view(1, x.shape[0], x.shape[1]).float()
    elif type == 'l':
        return torch.from_numpy(x).view(1, x.shape[0], x.shape[1]).long()
    else:
        raise NotImplementedError


def l2m(shape, lm, sigma=1.5):
    # lm = (x,y)
    m = np.zeros(shape, dtype=np.uint8)
    lm = lm.squeeze()
    lm[0] = int(np.round(lm[0]))
    lm[1] = int(np.round(lm[1]))

    if np.all(lm > 0) and lm[0] < shape[1] and lm[1] < shape[0]:
        x, y = np.meshgrid(np.linspace(-0.5, 0.5, m.shape[1]), np.linspace(-0.5, 0.5, m.shape[0]))
        mux = (lm[0] - m.shape[1] // 2) / 1. / m.shape[1]
        muy = (lm[1] - m.shape[0] // 2) / 1. / m.shape[0]
        s = sigma / 1. / m.shape[0]
        m = (x - mux) ** 2 / 2. / s ** 2 + (y - muy) ** 2 / 2. / s ** 2
        m = np.exp(-m)
        m -= m.min()
        m /= m.max()
    return m


def get_landmarks_from_hm(pred_map, remap_shape, pad=2, threshold=0.9):
    res = []

    for i in range(pred_map.shape[0]):
        m = pred_map[i, :, :]

        m -= m.min()
        m /= m.max()
        m *= 255
        m = m.astype(np.uint8)
        m = cv2.resize(m, remap_shape)

        tmp = m.mean(0)
        tmp /= tmp.max()

        x = np.where(tmp > threshold)[0]  # coords
        ind = np.diff(x).argmax().astype(int)
        if ind == 0:
            x = int(np.median(x))
        else:
            x = int(np.median(x[:ind]))  # leftmost cluster
        tmp = m[:, x - pad:x + pad].mean(1)

        tmp[np.isnan(tmp)] = 0
        tmp /= tmp.max()
        y = np.where(tmp > threshold)  #
        y = y[0][0]
        res.append([x, y])

    return np.array(res)


def wrap_img_landmarks_solt(img, landmarks):
    if not isinstance(img, np.ndarray):
        raise TypeError

    if not isinstance(landmarks, np.ndarray):
        raise TypeError

    if len(img.shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    if len(img.shape) != 3 or landmarks.shape[1] != 2:
        raise ValueError

    return sld.DataContainer((img, sld.KeyPoints(landmarks, img.shape[0], img.shape[1])), fmt='IP')


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


def solt_to_hourgalss_gs_input(dc: sld.DataContainer, downsample=4, sigma=1.5):
    if dc.data_format != 'IP':
        raise TypeError

    img, landmarks = dc.data

    landmarks_hm = l2m((img.shape[0] // downsample, img.shape[1] // downsample), landmarks.data // downsample, sigma)
    return img, landmarks_hm


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


def load_models(config, snapshots, device, side):
    """Loads models generated by model_getter into list.

    Useful when dealing with an ensemble.

    Parameters
    ----------
    model_getter : function or class
        returns a model object, into which we will load the state
    snapshots : list or tuple
        full paths to snapshots
    device : str
        Where to load the model: cpu or cuda?

    Returns
    -------
    out : list
        set of objects with loaded states and switched into eval mode.

    """
    tmp = []
    model_class = eval(config.model.name)
    for snp in snapshots:
        net = model_class.load_from_checkpoint(checkpoint_path=snp, config=config)

        net.freeze()
        tmp.append(net.to(device))
    return tmp


def create_model_from_conf(conf):
    model_class = eval(conf.model.name)
    model = model_class(config=conf)
    return model


def get_optimizer(model, opt_conf, classifier_only=False):
    if classifier_only and hasattr(model, 'classifier'):
        params = model.classifier.parameters()
    else:
        params = model.parameters()
    if opt_conf.name == 'Adam':
        optimizer = torch.optim.Adam(params=params, **opt_conf.params)
    elif opt_conf.name == 'SGD':
        optimizer = torch.optim.SGD(params=params, **opt_conf.params)
    else:
        raise ValueError('un processed optimizer, please define how you want to get your optimizer in '
                         f'{__file__}')
    return optimizer

def apply_fixed_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def apply_deterministic_computing(deterministic):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def get_snapshots(folder):
    snapshot_list = list()
    for root, dirs, files in os.walk(folder):
        for file in files:
            if '.ckpt' in file:
                snapshot_list.append(os.path.join(root, file))
    return snapshot_list


def create_roi_img(entry, img, spacing, cx, cy):
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
    cx += pad // 2
    cy += pad // 2

    size_px = int(np.round(roi_size / spacing))
    cy += int(np.round(pad_top / spacing))

    img_roi = img_orig_padded[int(math.fabs(cy - size_px // 2)):int(math.fabs(cy + size_px // 2)),
              int(math.fabs(cx - size_px // 2)):int(math.fabs(cx + size_px // 2))]
    return img_roi


def create_roi_img_with_points(entry, img, spacing, cx, cy, points):
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
    cx += pad // 2
    cy += pad // 2
    points += pad //2

    size_px = int(np.round(roi_size / spacing))
    cy += int(np.round(pad_top / spacing))
    # points = points.astype(np.int32)
    tmp_points = copy.copy(points)
    tmp_points[:, 0] -= int(math.fabs(cx - size_px // 2))
    tmp_points[:, 1] -= int(math.fabs(cy - size_px // 2))
    img_roi = img_orig_padded[int(math.fabs(cy - size_px // 2)):int(math.fabs(cy + size_px // 2)),
              int(math.fabs(cx - size_px // 2)):int(math.fabs(cx + size_px // 2))]
    tmp_points[:, 0] /= img_roi.shape[1]
    tmp_points[:, 1] /= img_roi.shape[0]

    return img_roi, tmp_points


def rotate_image(image, angle, center=None, scale=1.0):
    # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def rotate_point(p, origin=(0, 0), degrees=0):
    x, y = p
    cx, cy = origin
    rx = ((x - cx)*np.cos(degrees)) - ((y - cy) * np.sin(degrees)) + cx
    ry = ((x - cx)*np.sin(degrees)) + ((y - cy) * np.cos(degrees)) + cy
    return rx, ry


def plot_matrix_seaborn(matrix, labels, file_to_save):
    df_cm = pd.DataFrame(matrix, columns=labels, index=labels)
    # plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(9,8))
    sn.set(font_scale=1.2)

    sn.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={'size':10})
    if file_to_save is not None:
        plt.savefig(file_to_save, dpi=300)
    else:
        plt.show()
    plt.clf()


def plot_matrix_blue_shades(matrix, labels, file_to_save, ts='testset1'):
    h, w = matrix.shape
    clist = ['#E1F5FE', '#B3E5FC', '#81D4FA', '#4FC3F7', '#29B6F6', '#03A9F4', '#039BE5', '#0288D1', '#0277BD', '#01579B']
    # clist.reverse()
    cmap = colors.ListedColormap(clist)
    x_start = -0.5
    x_end = w - 0.5
    y_start = -0.5
    y_end = h - 0.5
    extent = [x_end, x_start, y_end, y_start]
    if ts == 'testset2':
        bounds = [10 + i * 3 for i in range(30)]
    else:
        bounds = [52 + i * 3 for i in range(17)]
    ticks = np.arange(0, h, 1)
    minor_ticks = np.arange(-0.5, h, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix*100, cmap=cmap, norm=norm)
    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=2)
    ax.tick_params(axis='x', which='major', labelsize=12, rotation=90)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    ax.xaxis.set_ticks_position('top')

    jump_x = (x_end - x_start) / (2.0 * w)
    jump_y = (y_end - y_start) / (2.0 * h)
    linx = np.linspace(start=x_start, stop=x_end, num=w, endpoint=False)
    liny = np.linspace(start=y_start, stop=y_end, num=h, endpoint=False)
    matrix = np.round(matrix, 2)
    for i, y in enumerate(liny):
        for j, x in enumerate(linx):
            if matrix[i,j] > 0.76:
                color = 'white'
            else:
                color = 'black'
            ax.text(x + jump_x, y + jump_y, str(matrix[i, j]), color=color, ha='center', va='center', size=8)
    cbar = fig.colorbar(im)
    cbar.set_ticks(bounds)
    bounds = np.asarray(bounds)
    bounds = np.round(bounds/100, 2)
    cbar.set_ticklabels(["%.2f"%b for b in bounds])
    cbar.ax.tick_params(axis='both', which='major', labelsize=8)
    if file_to_save is not None:
        plt.savefig(file_to_save, dpi=300)
    else:
        plt.show()
    plt.clf()


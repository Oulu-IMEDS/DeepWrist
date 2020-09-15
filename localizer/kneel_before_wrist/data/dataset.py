from torch.utils import data
import os
import cv2
import numpy as np
from localizer.kneel_before_wrist.data.utils import parse_landmarks
import solt.core as slc
import solt as slt
import solt.data as sld
ID_TO_SIDE = {0: 'PA', 1: 'LAT'}


class LandmarkDataset(data.Dataset):
    def __init__(self, data_root, split, hc_spacing, lc_spacing, transform, ann_type='hc', image_pad=100):
        if ann_type not in ['hc', 'lc']:
            raise ValueError('Wrong annotation type')

        self.img_pad = image_pad
        self.split = split
        self.transform = transform
        self.data_root = data_root
        self.ann_type = ann_type
        self.hc_spacing = hc_spacing
        self.lc_spacing = lc_spacing
        self.hc_lc_scale = self.hc_spacing / self.lc_spacing

    def __getitem__(self, index):
        subject_id, side, kl, t_lnd, f_lnd, _, center = self.split.iloc[index]
        kl = int(kl)

        if self.ann_type == 'hc':
            fname = os.path.join(self.data_root, f'{subject_id}_{kl}_{side}.png')
        else:
            fname = os.path.join(self.data_root, f'{subject_id}.png')

        img = cv2.imread(fname)

        if self.ann_type == 'hc':

            lndms = np.vstack((parse_landmarks(t_lnd), parse_landmarks(f_lnd)))
            kpts = slt.KeyPoints(lndms, img.shape[0], img.shape[1])
            dc = slc.DataContainer((img, kpts, kl), 'IPL')
        else:
            row, col, _ = img.shape
            center = np.array(list(map(int, center.split(',')))) * self.hc_lc_scale - self.hc_lc_scale * self.img_pad
            if side == 'L':
                img = img[:, col//2:]
                center[0] -= col//2
            else:
                img = img[:, :col // 2]

            center = slt.KeyPoints(np.expand_dims(center, 0), img.shape[0], img.shape[1])
            dc = slc.DataContainer((img, center, -1), 'IPL')

        transform_result = self.transform(dc)
        img, target_hm, target_kp, kl = transform_result

        res = {'img': img,
               'subject_id': subject_id,
               'kl': kl,
               'side': side,
               'kp_gt': target_kp}

        if target_hm is not None:
            res['target_hm'] = target_hm

        return res

    def __len__(self):
        return self.split.shape[0]


class LandmarkTrainListDataset(data.Dataset):
    def __init__(self, data_folder, meta, transform, side, return_dict=True):
        self.data_fld = data_folder
        self.meta = meta
        self.side = side
        self.transform = transform
        self.return_dict = return_dict

    def __getitem__(self, idx):
        entry = self.meta.iloc[idx]
        fname = entry.fname
        cx = entry.cx
        cy = entry.cy

        side = ID_TO_SIDE[self.side]
        file = os.path.join(self.data_fld, side, 'L',fname)
        if not os.path.isfile(file):
            file = os.path.join(self.data_fld, side, 'r', fname)
        img = cv2.imread(file, 0)
        # img = cv2.cvtColor(img,  cv2.COLOR_GRAY2RGB)
        lndm = np.array([cx, cy], dtype=np.float32).reshape((1, 2))
        kpts = sld.KeyPoints(lndm, img.shape[0], img.shape[1])
        dc = sld.DataContainer((img, kpts), 'IP')
        img,_,  target = self.transform(dc)
        if self.return_dict:
            return {'data': img, 'target': target, 'fname': fname, 'xy': (cx, cy)}
        else:
            return img, target

    def __len__(self):
        return self.meta.shape[0]


class MultipleLandmarkTrainListDataset(data.Dataset):
    def __init__(self, data_folder, meta, transform, side, return_dict=True):
        self.data_fld = data_folder
        self.meta = meta
        self.side = side
        self.transform = transform
        self.return_dict = return_dict

    def __getitem__(self, idx):
        entry = self.meta.iloc[idx]
        fname = entry.Fname
        points = eval(entry.Points)
        n = len(points)

        side = ID_TO_SIDE[self.side]
        file = os.path.join(self.data_fld, side,fname)

        img = cv2.imread(file, 0)
        # img = cv2.cvtColor(img,  cv2.COLOR_GRAY2RGB)
        lndm = np.array(points, dtype=np.float32).reshape((n, 2))
        kpts = sld.KeyPoints(lndm, img.shape[0], img.shape[1])
        dc = sld.DataContainer((img, kpts), 'IP')
        img,_,  target = self.transform(dc)
        if self.return_dict:
            return {'data': img, 'target': target, 'fname': fname, 'xy': points}
        else:
            return img, target

    def __len__(self):
        return self.meta.shape[0]
import os
import cv2
import torch.utils.data as data
import pandas as pd
import solt.data as sld
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

side_index = {0: 'PA', 1: 'LAT'}


class WristFractureDataset(data.Dataset):
    def __init__(self, root, meta, transform):
        self.root = root
        self.meta = meta
        self.transform = transform

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, i):
        entry = self.meta.iloc[i]
        label = entry.Fracture
        side = side_index[entry.Side]
        file_name = os.path.join(self.root, side, f'{entry.ID}_{side}.png')
        img = cv2.imread(file_name)
        dc = sld.DataContainer((img, label), 'IL')
        img, target = self.transform(dc)
        # img, label = self.transform((img, label))
        # return {'img': img, 'label': label, 'fname': entry.ID}
        return img, label




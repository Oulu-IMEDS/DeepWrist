import pickle
import time
from functools import partial

import torch
from sklearn.model_selection import GroupKFold
from termcolor import colored
from torchvision import transforms
import solt.transforms as slt
import solt.core as slc

from classifier.fracture_detector.data import apply_by_index, five_crop, wrap_img_target_solt, solt_to_img_target
from localizer.kneel_before_wrist.data.utils import solt2torchhm
import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import solt.data as sld
SIDE_DICT = {'PA': 0, 'LAT': 1}
IND_TO_SIDE = {0: 'PA', 1: 'LAT'}


class SplitDataToFunction(object):

    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x[0], x[1])

    def __repr__(self):
        return self.__class__.__name__ + '(function={0}'.format(self.func)


class DataToFunction(object):

    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x)

    def __repr__(self):
        return self.__class__.__name__ + '(function={0}'.format(self.func)


class ApplyByIndex(object):

    def __init__(self, func, index):
        self.func = func
        self.index = index

    def __call__(self, x):
        return apply_by_index(x, self.func, self.index)

    def __repr__(self):
        return self.__class__.__name__ + '(function={0}, index={1}'.format(self.func, self.index)


class ApplyCustomFunction(object):
    def __init__(self, func, params):
        self.function = func
        self.params = params

    def __call__(self, x):
        return self.function(x, **self.params)

    def __repr__(self):
        return self.__class__.__name__ + '(function={0}, index={1}'.format(self.func, self.index)


class StackFlippedImage(object):
    def __call__(self, x):
        flipped = cv2.flip(x, flipCode=1) # flip around y axis
        return np.stack((x, flipped))

    def __repr__(self):
        return self.__class__.__name__ +  'Flipping around y axis'


class FiveCroppedVStack(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x):
        return np.vstack(five_crop(I, self.crop_size) for I in x)


class TensoredStack(object):
    def __init__(self, norm):
        self.normalization = norm

    def __call__(self, x):
        return torch.stack([self.normalization(transforms.ToTensor()(crop)) for crop in x])


def get_wrist_fracture_transformation(crop_size):
    return transforms.Compose([
        SplitDataToFunction(wrap_img_target_solt),
        slc.Stream([
            slt.RandomFlip(p=1, axis=1),
            slt.RandomProjection(affine_transforms=slc.Stream([
                slt.RandomScale(range_x=(0.8, 1.2), p=1),
                slt.RandomShear(range_x=(-0.1, 0.1), p=0.5),
                slt.RandomShear(range_y=(-0.1, 0.1), p=0.5),
                slt.RandomRotate(rotation_range=(-10, 10), p=1),
            ]), v_range=(1e-5, 5e-4), p=0.8),
            slt.PadTransform(pad_to=(256, 256), padding='z'),
            slt.CropTransform(crop_size, crop_mode='r'),
            slc.SelectiveStream([
                slc.SelectiveStream([
                    slt.ImageSaltAndPepper(p=1, gain_range=0.01),
                    slt.ImageBlur(p=0.5, blur_type='m', k_size=(11,)),
                ]),
                slt.ImageAdditiveGaussianNoise(p=1, gain_range=0.5),
            ]),
            slt.ImageGammaCorrection(p=1, gamma_range=(0.5, 1.5)),
        ]),
        DataToFunction(solt_to_img_target),
        ApplyByIndex(transforms.ToTensor(), 0)
    ])


def get_test_meta(args):
    meta_test = pd.read_csv(os.path.join(args.dataset_home, 'test_split.csv'))
    print(colored('==> ', 'green')+f'Test set size for PA / # fractures: '
                                   f'{meta_test[meta_test.Side == 0].shape[0]} / '
                                   f'{meta_test[(meta_test.Side == 0) & (meta_test.Fracture == 1)].shape[0]}')

    print(colored('==> ', 'green') + f'Test set size for Lateral / # fractures: '
                                     f'{meta_test[meta_test.Side == 1].shape[0]} / '
                                     f'{meta_test[(meta_test.Side == 1) & (meta_test.Fracture == 1)].shape[0]}')
    return meta_test


def get_train_val_transformations(config, meta,  side):
    meta_train = meta[meta.Side == side]
    # calculate mean and std of the dataset
    mean_std_file = os.path.join(config.dataset.data_home, f'mean_std_{side}.npy')
    if os.path.isfile(mean_std_file):
        print(f'Using mean and std from {mean_std_file}')
        mean, std = np.load(mean_std_file)
    else:
        print('Mean STD not found, calculating from data')
        mean = np.zeros(3)
        std = np.zeros(3)
        pbar = tqdm(meta_train.shape[0])
        for index in range(meta_train.shape[0]):
            entry = meta_train.iloc[index]
            side_txt = IND_TO_SIDE[side]
            file_name = os.path.join(config.dataset.data_home,side_txt, f'{entry.ID}_{side_txt}.png')
            img = cv2.imread(file_name)
            if np.max(img) >= 1:
                img = np.asarray(img, dtype=np.float32) / 255.0
            mean += np.mean(img, axis=(0, 1))
            std += np.std(img, axis=(0, 1))
            pbar.update()
        pbar.close()
        mean /= meta_train.shape[0]
        std /= meta_train.shape[0]
        np.save(mean_std_file, [mean, std])
    # get transformation
    transformations = get_wrist_fracture_transformation(config.dataset.crop_size)
    normalization = transforms.Normalize(mean=mean, std=std)

    train_trf = transforms.Compose([transformations,
                                    ApplyByIndex(normalization, 0)])
    val_trf = transforms.Compose([ApplyByIndex(transforms.ToTensor(), 0),
                                  ApplyByIndex(normalization, 0)
                                  ])

    return train_trf, val_trf


def get_wr_tta(config, side):
    mean_std_file = os.path.join(str(config.dataset.train_data_home), f'mean_std_{side}.npy')
    mean, std = np.load(mean_std_file)
    normalization = transforms.Normalize(mean=mean, std=std)
    tta = transforms.Compose([
        ApplyCustomFunction(func=cv2.cvtColor, params={'code':cv2.COLOR_GRAY2RGB}),
        StackFlippedImage(),
        FiveCroppedVStack(crop_size=config.dataset.crop_size),
        TensoredStack(norm=normalization)
    ])
    return tta


def get_landmark_transform_kneel(config):
    cutout = slt.ImageCutOut(cutout_size=(int(config.dataset.cutout * config.dataset.augs.crop.crop_x),
                                          int(config.dataset.cutout * config.dataset.augs.crop.crop_y)),
                             p=0.5)
    ppl = transforms.Compose([
        slc.Stream(),
        slc.SelectiveStream([
            slc.Stream([
                slt.RandomFlip(p=0.5, axis=1),
                slt.RandomProjection(affine_transforms=slc.Stream([
                    slt.RandomScale(range_x=(0.9, 1.1), p=1),
                    slt.RandomRotate(rotation_range=(-90, 90), p=1),
                    slt.RandomShear(range_x=(-0.1, 0.1), range_y=(-0.1, 0.1), p=0.5),
                    slt.RandomShear(range_x=(-0.1, 0.1), range_y=(-0.1, 0.1), p=0.5),
                ]), v_range=(1e-5, 2e-3), p=0.5),
                # slt.RandomScale(range_x=(0.5, 2.5), p=0.5),
            ]),
            slc.Stream()
        ], probs=[0.7, 0.3]),
        slc.Stream([
            slt.PadTransform((config.dataset.augs.pad.pad_x, config.dataset.augs.pad.pad_y), padding='z'),
            slt.CropTransform((config.dataset.augs.crop.crop_x, config.dataset.augs.crop.crop_y), crop_mode='r'),
        ]),
        slc.SelectiveStream([
            slt.ImageSaltAndPepper(p=1, gain_range=0.01),
            slt.ImageBlur(p=1, blur_type='g', k_size=(3, 5)),
            slt.ImageBlur(p=1, blur_type='m', k_size=(3, 5)),
            slt.ImageAdditiveGaussianNoise(p=1, gain_range=0.5),
            slc.Stream([
                slt.ImageSaltAndPepper(p=1, gain_range=0.05),
                slt.ImageBlur(p=0.5, blur_type='m', k_size=(3, 5)),
            ]),
            slc.Stream([
                slt.ImageBlur(p=0.5, blur_type='m', k_size=(3, 5)),
                slt.ImageSaltAndPepper(p=1, gain_range=0.01),
            ]),
            slc.Stream()
        ], n=1),
        slt.ImageGammaCorrection(p=0.5, gamma_range=(0.5, 1.5)),
        cutout if config.dataset.use_cutout else slc.Stream(),
        DataToFunction(solt_to_img_target),
        ApplyByIndex(transforms.ToTensor(), 0)
    ])
    return ppl


def get_train_val_transformations_kneel(config, meta,  side):
    meta_train = meta[meta.Side == side]
    # calculate mean and std of the dataset
    mean_std_file = os.path.join(config.dataset.data_home, f'mean_std_{side}.npy')
    if os.path.isfile(mean_std_file):
        print(f'Using mean and std from {mean_std_file}')
        mean, std = np.load(mean_std_file)
    else:
        print('Mean STD not found, calculating from data')
        mean = np.zeros(3)
        std = np.zeros(3)
        pbar = tqdm(meta_train.shape[0])
        for index in range(meta_train.shape[0]):
            entry = meta_train.iloc[index]
            side_txt = IND_TO_SIDE[side]
            file_name = os.path.join(config.dataset.data_home,side_txt, f'{entry.ID}_{side_txt}.png')
            img = cv2.imread(file_name)
            if np.max(img) >= 1:
                img = np.asarray(img, dtype=np.float32) / 255.0
            mean += np.mean(img, axis=(0, 1))
            std += np.std(img, axis=(0, 1))
            pbar.update()
        pbar.close()
        mean /= meta_train.shape[0]
        std /= meta_train.shape[0]
        np.save(mean_std_file, [mean, std])
    # get transformation
    transformations = get_landmark_transform_kneel(config)
    normalization = transforms.Normalize(mean=mean, std=std)

    train_trf = transforms.Compose([transformations,
                                    ApplyByIndex(normalization, 0)])
    val_trf = transforms.Compose([ApplyByIndex(transforms.ToTensor(), 0),
                                  ApplyByIndex(normalization, 0)
                                  ])

    return train_trf, val_trf
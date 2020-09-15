import os
from functools import partial

from termcolor import colored
from torch.utils.data import DataLoader
import cv2
import utils as dwutils
from torchvision import transforms
import solt.core as slc
import solt.transforms as slt
import numpy as np
from tqdm import tqdm
from localizer.kneel_before_wrist.data.dataset import LandmarkTrainListDataset, MultipleLandmarkTrainListDataset
from localizer.kneel_before_wrist.data.utils import solt2torchhm
from utils.transformation import ColorPaddingWithSide, SIDES, TriangularMask, LowVisibilityTransform, SubSampleUpScale


class WrapImageLandmarksSOLT(object):

    def __call__(self, data):
        """
        Parameter:
        ----------
        data: image and landmarks
            image data and landmark to be converted to solt object

        Returns:
        --------
        solt object: wrapped solt object.
        """
        return dwutils.wrap_img_landmarks_solt(data[0], data[1])

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SOLTtoHourGlassGSinput(object):
    def __init__(self, downsample, sigma):
        self.downsample = downsample
        self.sigma = sigma

    def __call__(self, data):
        return dwutils.solt_to_hourgalss_gs_input(data, downsample=self.downsample, sigma=self.sigma)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ApplyTransformByIndex(object):
    def __init__(self, transform, ids):
        self.transform = transform
        self.ids = ids

    def __call__(self, data):
        return dwutils.apply_by_index(data=data, transform=self.transform, idx=self.ids)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ApplyCustomFunction(object):
    def __init__(self, func, params):
        self.function = func
        self.params = params

    def __call__(self, x):
        return self.function(x, **self.params)

    def __repr__(self):
        return self.__class__.__name__ + '(function={0}, index={1}'.format(self.func, self.index)


def get_landmark_transform_kneel(config):
    cutout = slt.ImageCutOut(cutout_size=(int(config.dataset.cutout * config.dataset.augs.crop.crop_x),
                                          int(config.dataset.cutout * config.dataset.augs.crop.crop_y)),
                             p=0.5)
    # plus-minus 1.3 pixels
    jitter = slt.KeypointsJitter(dx_range=(-0.003, 0.003), dy_range=(-0.003, 0.003))
    ppl = transforms.Compose([
        ColorPaddingWithSide(p=0.05, pad_size=10, side=SIDES.RANDOM, color=(50,100)),
        TriangularMask(p=0.025, arm_lengths=(100, 50), side=SIDES.RANDOM, color=(50,100)),
        TriangularMask(p=0.025, arm_lengths=(50, 100), side=SIDES.RANDOM, color=(50,100)),
        LowVisibilityTransform(p=0.05, alpha=0.15, bgcolor=(50,100)),
        SubSampleUpScale(p=0.01),
        jitter if config.dataset.augs.use_target_jitter else slc.Stream(),
        slc.SelectiveStream([
            slc.Stream([
                slt.RandomFlip(p=0.5, axis=1),
                slt.RandomProjection(affine_transforms=slc.Stream([
                    slt.RandomScale(range_x=(0.9, 1.1), p=1),
                    slt.RandomRotate(rotation_range=(-90, 90), p=1),
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
        partial(solt2torchhm, downsample=None, sigma=None),
    ])
    return ppl


def get_landmark_transform(config):
    return transforms.Compose([
        # WrapImageLandmarksSOLT(),
        slc.Stream([
            slt.RandomFlip(p=0.5, axis=1),
            slt.RandomScale(range_x=(0.8, 1.2), p=1),
            slt.RandomRotate(rotation_range=(-180, 180), p=0.2),
            slt.RandomProjection(affine_transforms=slc.Stream([
                slt.RandomScale(range_x=(0.8, 1.3), p=1),
                slt.RandomRotate(rotation_range=(-180, 180), p=1),
                slt.RandomShear(range_x=(-0.1, 0.1), range_y=(0, 0), p=0.5),
                slt.RandomShear(range_y=(-0.1, 0.1), range_x=(0, 0), p=0.5),
            ]), v_range=(1e-5, 2e-3), p=0.8),
            slt.PadTransform(int(config.dataset.crop_size * 1.4), padding='z'),
            slt.CropTransform(config.dataset.crop_size, crop_mode='r'),
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
            ]),
            slt.ImageGammaCorrection(p=1, gamma_range=(0.5, 1.5))
        ]),
        SOLTtoHourGlassGSinput(downsample=4, sigma=3),
        ApplyTransformByIndex(transform=dwutils.npg2tens, ids=[0, 1]),
    ])


def calculate_mean_std_from_dataset(train_meta, config, file, trf, side=0):
    # Mean estimation
    data_folder = os.path.join(config.dataset.data_home, config.dataset.data_folder)
    train_ds = MultipleLandmarkTrainListDataset(data_folder, train_meta, transform=trf,
                                        side=side)
    train_loader = DataLoader(train_ds, batch_size=config.train_params.train_bs,
                              num_workers=config.dataset.n_data_workers)

    mean_vector = np.zeros(1)
    std_vector = np.zeros(1)

    print(colored('==> ', 'green') + 'Estimating the mean')
    pbar = tqdm(total=len(train_loader))
    for entry in train_loader:
        batch = entry['data']
        for j in range(mean_vector.shape[0]):
            mean_vector[j] += batch[:, j, :, :].mean()
            std_vector[j] += batch[:, j, :, :].std()
        pbar.update()
    mean_vector /= len(train_loader)
    std_vector /= len(train_loader)
    np.save(file, [mean_vector, std_vector])
    pbar.close()
    return mean_vector, std_vector


def get_train_val_transform(landmark_trf, normalization, config):
    train_transform = transforms.Compose([
        landmark_trf,
        lambda x: dwutils.apply_by_index(x, normalization, 0),
    ])

    val_transform = transforms.Compose([
        landmark_trf,
        # slc.Stream([
        #     slt.PadTransform(config.dataset.crop_size, padding='z'),
        #     slt.CropTransform(config.dataset.crop_size, crop_mode='c'),
        # ]),
        lambda x: dwutils.apply_by_index(x, normalization, 0),
    ])
    return train_transform, val_transform
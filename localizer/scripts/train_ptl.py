import os
import tempfile
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import ListConfig, OmegaConf
import yaml
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from classifier.config import get_conf
from torchvision import transforms
import torch

from localizer.kneel_before_wrist.data.dataset import  MultipleLandmarkTrainListDataset
from localizer.kneel_before_wrist.data.transforms import get_landmark_transform_kneel, calculate_mean_std_from_dataset, \
    get_train_val_transform
from utils import apply_fixed_seed, apply_deterministic_computing

# do not remove
from localizer.kneel_before_wrist.model import HourglassNet_PTL

IND_TO_SIDE = {0: 'PA', 1: 'LAT'}

if __name__ == '__main__':
    cwd = Path().cwd()
    conf_file = cwd.parents[0] / 'config' / 'config.yaml'
    config = get_conf(conf_file=conf_file, cwd=cwd)

    # save config
    os.makedirs(config.snapshot_dir, exist_ok=True)
    OmegaConf.save(config=config, f=os.path.join(config.snapshot_dir, 'params.yaml'))

    apply_fixed_seed(config.seed)
    apply_deterministic_computing(config.deterministic)

    meta_path = os.path.join(config.dataset.data_home, config.dataset.data_folder, config.dataset.meta)
    # master copy of meta data
    master_meta = pd.read_csv(meta_path)
    # check the number of sides
    if isinstance(config.dataset.side, int):
        config.dataset.side = [config.dataset.side]

    for side in config.dataset.side:
        train_meta = master_meta[master_meta.Side == side]
        landmark_trf = get_landmark_transform_kneel(config)
        # get mean and std
        ms_file = os.path.join(config.dataset.data_home, f'mean_std_{side}.npy')
        if os.path.isfile(ms_file):
            mean, std = np.load(ms_file)
        else:
            mean, std = calculate_mean_std_from_dataset(train_meta, config, ms_file, side=side, trf=landmark_trf)

        normalization = transforms.Compose([
            transforms.Normalize(torch.from_numpy(mean).float(), torch.from_numpy(std).float())
        ])
        train_transform, val_transform = get_train_val_transform(landmark_trf, normalization, config)

        for fold, (train_ind, val_ind) in enumerate(KFold(config.train_params.n_fold).split(train_meta)):
            snapshot_dir = os.path.join(config.snapshot_dir, IND_TO_SIDE[side], f'fold_{fold}')

            train_ds = MultipleLandmarkTrainListDataset(
                data_folder=os.path.join(config.dataset.data_home, config.dataset.data_folder),
                meta=train_meta.iloc[train_ind], transform=train_transform,
                side=side, return_dict=False)
            val_ds = MultipleLandmarkTrainListDataset(
                data_folder=os.path.join(config.dataset.data_home, config.dataset.data_folder),
                meta=train_meta.iloc[val_ind], transform=val_transform,
                side=side, return_dict=False)
            train_loader = DataLoader(dataset=train_ds,
                                      batch_size=config.train_params.train_bs,
                                      num_workers=config.dataset.n_data_workers, drop_last=False,
                                      shuffle=True,
                                      pin_memory=True)

            val_loader = DataLoader(dataset=val_ds,
                                    batch_size=config.train_params.val_bs,
                                    num_workers=config.dataset.n_data_workers,
                                    shuffle=False,
                                    pin_memory=True)

            model_class = eval(config.model.name)
            model = model_class(config)
            os.makedirs(snapshot_dir, exist_ok=True)
            conf_file = os.path.join(snapshot_dir, f'config.yaml')
            with open(conf_file, 'w') as f:
                yaml.dump(config, f)
            trainer = pl.Trainer(gpus=[config.local_rank], max_epochs=config.train_params.n_epochs, default_save_path=snapshot_dir)
            trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
            torch.cuda.empty_cache()

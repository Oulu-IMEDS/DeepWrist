import pickle
from pathlib import Path

import torch
import os

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from classifier.config import get_conf
from classifier.fracture_detector.data import get_meta, WristFractureDataset
from classifier.fracture_detector.data._transform import get_train_val_transformations_kneel
from classifier.fracture_detector.model import ModelWithTemperature
from utils import apply_fixed_seed, apply_deterministic_computing, get_snapshots, FractureDetector

if __name__ == '__main__':
    cwd = Path().cwd()
    conf_file = cwd.parents[0] / 'config' / 'config.yaml'
    config = get_conf(conf_file=conf_file, cwd=cwd)

    apply_fixed_seed(config.seed)
    apply_deterministic_computing(config.deterministic)

    if isinstance(config.local_rank, int):
        device = torch.device(f'cuda:{config.local_rank}')
        torch.cuda.set_device(config.local_rank)
    else:
        device = torch.device('cpu')
    # meta is the master meta here
    meta = get_meta(config)

    if isinstance(config.dataset.side, int):
        config.dataset.side = [config.dataset.side]
    fd_lat_folder = os.path.join(config.snapshot_folder, 'LAT')
    fd_pa_folder = os.path.join(config.snapshot_folder, 'PA')
    fd_lat_snapshots = get_snapshots(fd_lat_folder)
    fd_pa_snapshots = get_snapshots(fd_pa_folder)
    lat_detector = FractureDetector(config, fd_lat_snapshots, side=1, device=device)
    pa_detector = FractureDetector(config, fd_pa_snapshots, side=0, device=device)
    meta_pa = meta[meta.Side == 0]
    meta_lat = meta[meta.Side == 1]
    _, pa_trf = get_train_val_transformations_kneel(config, meta, 0)
    _, lat_trf = get_train_val_transformations_kneel(config, meta, 1)

    gkf = GroupKFold(5)
    _, val_ind_pa = next(gkf.split(meta_pa, meta_pa.Fracture, meta_pa.ID))
    gkf = GroupKFold(5) # gfk need to re-initialize to have the same validaiton data as the training
    _, val_ind_lat = next(gkf.split(meta_lat, meta_lat.Fracture, meta_lat.ID))
    val_ds_pa = WristFractureDataset(root=config.dataset.data_home, meta=meta_pa.iloc[val_ind_pa],
                         transform=pa_trf)

    val_ds_lat = WristFractureDataset(root=config.dataset.data_home, meta=meta_lat.iloc[val_ind_lat],
                         transform=lat_trf)

    loader_pa = DataLoader(dataset=val_ds_pa,
                              batch_size=config.train_params.val_bs,
                              num_workers=config.dataset.n_data_workers,
                              shuffle=False,
                              pin_memory=True)

    loader_lat = DataLoader(dataset=val_ds_lat,
                            batch_size=config.train_params.val_bs,
                            num_workers=config.dataset.n_data_workers,
                            shuffle=False,
                            pin_memory=True)
    temp_dict = dict()
    temp_dict['PA'] = list()
    temp_dict['LAT'] = list()
    for model in pa_detector.models:
        model_with_tmp = ModelWithTemperature(model, device)
        model_with_tmp.set_temperature(loader_pa)
        temp_dict['PA'].append(model_with_tmp.temperature.item())

    for model in lat_detector.models:
        model_with_tmp = ModelWithTemperature(model, device)
        model_with_tmp.set_temperature(loader_lat)
        temp_dict['LAT'].append(model_with_tmp.temperature.item())
    with open('temp_old.pkl', 'wb') as f:
        pickle.dump(temp_dict, f)
    print(temp_dict)




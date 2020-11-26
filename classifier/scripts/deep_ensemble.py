import os
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateLogger
import torch
from classifier.config import get_conf
from classifier.fracture_detector.callback import ReleaseAfterCallback
from classifier.fracture_detector.data import get_train_val_transformations, WristFractureDataset, get_meta
from classifier.fracture_detector.data._transform import get_train_val_transformations_kneel
from utils import create_model_from_conf, apply_fixed_seed, apply_deterministic_computing
IND_TO_SIDE = {0: 'PA', 1: 'LAT'}

if __name__ == '__main__':
    cwd = Path().cwd()
    conf_file = cwd.parents[0] / 'config' / 'config.yaml'
    config = get_conf(conf_file=conf_file, cwd=cwd)

    # save configs
    os.makedirs(config.snapshot_dir, exist_ok=True)
    OmegaConf.save(config=config, f=os.path.join(config.snapshot_dir, 'params.yaml'))
    # fixed seed and deterministic environment
    apply_fixed_seed(config.seed)
    apply_deterministic_computing(config.deterministic)
    path_dict = {
        'original_cwd': cwd.parents[0],
        'workdir': Path(config.workdir),
        'data_dir': Path(config.workdir) / 'data',
        'checkpoint_path': Path(config.snapshot_dir)
    }
    # meta is the master meta here
    meta = get_meta(config)

    if isinstance(config.dataset.side, int):
        config.dataset.side = [config.dataset.side]
    torch.cuda.set_device(config.local_rank)
    for side in config.dataset.side:
        train_meta = meta[meta.Side == side]
        train_trf, val_trf = get_train_val_transformations_kneel(config, meta, side)
        # train_df, val_df = train_test_split(train_meta, test_size=config.dataset.val, shuffle=True)
        gkf = GroupKFold(5)
        train_ind, val_ind = next(gkf.split(train_meta, train_meta.Fracture, train_meta.ID))

        train_ds = WristFractureDataset(root=config.dataset.data_home, meta=train_meta.iloc[train_ind],
                                        transform=train_trf)
        val_ds = WristFractureDataset(root=config.dataset.data_home, meta=train_meta.iloc[val_ind],
                                      transform=val_trf)
        train_loader = DataLoader(dataset=train_ds,
                                  batch_size=config.train_params.train_bs,
                                  num_workers=config.dataset.n_data_workers, drop_last=False,
                                  shuffle=True)

        val_loader = DataLoader(dataset=val_ds,
                                batch_size=config.train_params.val_bs,
                                num_workers=config.dataset.n_data_workers,
                                shuffle=False)

        for ens in range(config.train_params.num_ensemble):
            # start with differnt seeds
            new_seed = config.seed * (ens + 1)
            apply_fixed_seed(new_seed)
            snapshot_dir = os.path.join(config.snapshot_dir, IND_TO_SIDE[side], f'model_{ens}')

            model = create_model_from_conf(config)
            lr_cb = LearningRateLogger()
            callback = ReleaseAfterCallback(config.train_params.release_after)
            trainer = pl.Trainer(gpus=[config.local_rank], auto_select_gpus=False, callbacks=[lr_cb, callback], max_epochs=config.train_params.n_epochs,
                                 default_save_path=snapshot_dir)
            trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)



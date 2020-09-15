from omegaconf import OmegaConf
from time import localtime, strftime
from pathlib import Path


def get_conf(cwd, conf_file):
    OmegaConf.register_resolver('now', lambda fmt: strftime(fmt, localtime()))
    config = OmegaConf.load(str(conf_file))
    conf_cli = OmegaConf.from_cli()
    for entry in config.defaults:
        assert len(entry) == 1
        for k, v in entry.items():
            if k in conf_cli:
                v = conf_cli[k]
            entry_path = cwd.parents[0] / 'config' / k / f'{v}.yaml'
            entry_conf = OmegaConf.load(str(entry_path))
            config = OmegaConf.merge(config, entry_conf)
    config = OmegaConf.merge(config, conf_cli)
    return config
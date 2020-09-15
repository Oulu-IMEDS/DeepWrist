import torch
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import utils._utils as dwutils
from localizer.kneel_before_wrist.model import HourglassNet_PTL


class LandmarkAnnotator:
    """Annotates landmarks

    """
    device = torch.device('cuda:0')

    def __init__(self, config, snapshots, side):

        mean_vector, std_vector = np.load(os.path.join(config.dataset.train_data_home, f'mean_std_{side}.npy'))
        norm_transform = transforms.Normalize(torch.from_numpy(mean_vector).float(),
                                              torch.from_numpy(std_vector).float())

        self.inference_transform = transforms.Compose([
            dwutils.npg2tens,
            norm_transform
        ])

        # Loading the models
        self.models = dwutils.load_models(config, snapshots, self.device, side)

    def annotate(self, img):
        img = self.inference_transform(img).unsqueeze(0).to(self.device)
        lndm_hm = None
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if lndm_hm is None:
                    lndm_hm = model(img)
                else:
                    lndm_hm += model(img)

        lndm_hm /= len(self.models)
        lndm_hm = lndm_hm.detach().cpu().numpy().squeeze()
        cx = lndm_hm[0]
        cy = lndm_hm[1]
        return lndm_hm

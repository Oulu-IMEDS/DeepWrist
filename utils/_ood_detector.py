import numpy as np
import torch
from torchvision import transforms
import os
import torch.nn.functional as F
import cv2
from ._utils import load_models
from classifier.fracture_detector.data import get_wr_tta
# for debug
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss


class OODDetector(object):
    def __init__(self, config, snapshots, side, device, temp, trf):
        self.device = device
        self.temp = temp
        self.inference_transform = trf
        # Loading the models
        self.models = load_models(config, snapshots, self.device, side=side)
        self.config = config

    def detect(self, img, label):
        h, w = img.shape
        # img = np.expand_dims(img, axis=-1) # add channel dim
        img = cv2.cvtColor(img, code=cv2.COLOR_GRAY2RGB)
        img, _ = self.inference_transform([img, label])
        img = img.to(self.device)
        img = img.unsqueeze(0)
        n, c, _, _ = img.shape
        preds = list()
        nlls = list()
        briers = list()
        entropies = list()
        sms = list()
        for i, model in enumerate(self.models):
            # T = self.temp[i]
            T = 1.0
            model.eval()
            with torch.no_grad():
                logits = model(img)
                sm = torch.softmax(logits / T, dim=1)
                sm = sm.detach().cpu().numpy()
                sms.append(sm[0])
                pred = sm[:, 1]
                nll = -np.log(sm[:, label])
                entropy = -np.sum(np.log(sm + 1e-9) * sm, axis=1)
                bs = brier_score_loss(y_true=np.asarray([label]*len(pred)), y_prob=pred.reshape(-1))

                nlls.append(nll[0])
                briers.append(bs)
                preds.append(pred[0])
                entropies.append(entropy[0])

        return preds, nlls, briers, entropies, sms
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



class FractureDetector(object):
    device = 'cuda:0'

    def __init__(self, config, snapshots, side):

        self.inference_transform = get_wr_tta(config, side=side)
        # Loading the models
        self.models = load_models(config, snapshots, self.device, side=side)
        self.config = config

    def detect(self, img):
        h, w = img.shape
        img = self.inference_transform(img).to(self.device)
        n, c, _, _ = img.shape
        preds = list()
        gcams_list = list()
        T = 1.0
        for model in self.models:
            model.eval()
            with torch.no_grad():
                acts = model.encoder[:-1](img)
                features = model.encoder[-1](acts).view(n, -1)
            with torch.enable_grad():
                grads = []
                features.requires_grad = True
                features.register_hook(lambda g: grads.append(g))
                logits = model.classifier(features)

                sm = torch.softmax(logits / T, dim=1)
                grad_y = torch.zeros_like(sm)
                grad_y[:, 1] = 1
                sm.backward(grad_y)
                pred = sm[:, 1].detach().cpu().numpy()
                preds.append(pred.mean())
            grads = grads[0]
            grads = grads.unsqueeze(-1)
            grads = grads.unsqueeze(-1)
            grads = grads * acts
            grads = grads.sum(1)
            grads = F.relu(grads)
            grads = grads.unsqueeze(1)
            # gcams = F.relu((grads[0].unsqueeze(-1).unsqueeze(-1) * acts).sum(1)).unsqueeze(1)
            gcams = F.interpolate(grads, size=(img.shape[-2], img.shape[-1]), mode='bilinear', align_corners=True).to('cpu').squeeze().numpy()
            gcams_list.append(gcams)
        gcams = np.mean(gcams_list, axis=0)
        gcams_h, gcams_w = gcams[0].shape
        border_z = 15
        mask_gcam = np.zeros((gcams_h, gcams_w))
        mask_gcam[border_z:-border_z, border_z:-border_z] = np.ones((gcams_h - 2 * border_z, gcams_w - 2 * border_z))
        mask_gcam = cv2.GaussianBlur(mask_gcam, (5, 5), 25)

        gcams = [gcam * mask_gcam for gcam in gcams]

        heatmap = np.zeros((h, w))

        size = self.config.dataset.crop_size
        x1 = w // 2 - size // 2
        y1 = h // 2 - size // 2
        # gradcam + TTA: flips and 5-crop

        # upper-left crop
        heatmap[0:size, 0:size] += gcams[1]
        heatmap[0:size, 0:size] += cv2.flip(gcams[6], 1)

        # upper-right crop
        heatmap[0:size, w - size:w] += gcams[2]
        heatmap[0:size, w - size:w] += cv2.flip(gcams[7], 1)

        # bottom-left crop
        heatmap[h - size:h, 0:size] += gcams[3]
        heatmap[h - size:h, 0:size] += cv2.flip(gcams[8], 1)

        heatmap[h - size:h, w - size:w] += gcams[4]
        heatmap[h - size:h, w - size:w] += cv2.flip(gcams[9], 1)
        # Center crop
        heatmap[y1:y1 + size, x1:x1 + size] += gcams[0]
        heatmap[y1:y1 + size, x1:x1 + size] += cv2.flip(gcams[5], 1)

        heatmap -= heatmap.min()
        if heatmap.max() != 0:
            heatmap /= heatmap.max()
        final_pred = np.mean(preds)
        return final_pred, heatmap
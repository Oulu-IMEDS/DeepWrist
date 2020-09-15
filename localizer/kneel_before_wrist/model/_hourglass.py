import torch
from sklearn.metrics import accuracy_score
from torch import nn
from deeppipeline.common.modules import conv_block_1x1
from deeppipeline.keypoints.models.modules import Hourglass, HGResidual, MultiScaleHGResidual, SoftArgmax2D
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class HourglassNet_PTL(pl.LightningModule):
    def __init__(self, config):
        super(HourglassNet_PTL, self).__init__()
        self.config = config
        self.params = config.model.params
        self.criterion = self.get_loss(config.loss)
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.params.n_inputs, self.params.bw, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.params.bw),
            nn.ReLU(inplace=True),
            self.__make_hg_block(self.params.bw, self.params.bw * 2),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            self.__make_hg_block(self.params.bw * 2, self.params.bw * 2),
            self.__make_hg_block(self.params.bw * 2, self.params.bw * 2),
            self.__make_hg_block(self.params.bw * 2, self.params.bw * 4)
        )

        self.hourglass = Hourglass(n=self.params.hg_depth, hg_width=self.params.bw * 4, n_inp=self.params.bw * 4,
                                   n_out=self.params.bw * 8, upmode=self.params.upmode,
                                   multiscale_block=self.params.multiscale_hg_block, se=self.params.se,
                                   se_ratio=self.params.se_ratio)

        self.mixer = nn.Sequential(nn.Dropout2d(p=0.25),
                                   conv_block_1x1(self.params.bw * 8, self.params.bw * 8),
                                   nn.Dropout2d(p=0.25),
                                   conv_block_1x1(self.params.bw * 8, self.params.bw * 4))

        self.out_block = nn.Sequential(nn.Conv2d(self.params.bw * 4, self.params.n_outputs, kernel_size=1, padding=0))
        self.sagm = SoftArgmax2D()

    def __make_hg_block(self, inp, out):
        if self.params.multiscale_hg_block:
            return MultiScaleHGResidual(inp, out, se=self.params.se, se_ratio=self.params.se_ratio)
        else:
            return HGResidual(inp, out, se=self.params.se, se_ratio=self.params.se_ratio)

    def forward(self, x):
        o_layer_1 = self.layer1(x)
        o_layer_2 = self.layer2(o_layer_1)

        o_hg = self.hourglass(o_layer_2)
        o_mixer = self.mixer(o_hg)
        out = self.out_block(o_mixer)

        return self.sagm(out)

    def get_loss(self, params):
        loss_class = eval(params.class_name)
        criterion = loss_class().to(self.config.local_rank)
        return criterion

    def training_step(self, batch, batch_nb, optimizer_idx=0):
        x, y = batch
        if self.config.train_params.mixup.enabled:
            inputs, targets_a, targets_b, lam, = mixup_data(x, y, self.config.train_params.mixup.alpha)
            pred = self(inputs)
            loss = lam * self.criterion(pred, targets_a) + (1- lam) * self.criterion(pred, targets_b)
        else:
            loss = self.criterion(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        log = {'val_loss': loss}
        return {'val_loss': loss,  'log': log}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        optim_class = eval(self.config.optimizer.class_name)
        optim = optim_class(self.parameters(), **self.config.optimizer.params)
        scheduler_class = eval(self.config.optimizer.scheduler.class_name)
        scheduler = scheduler_class(optim, **self.config.optimizer.scheduler.params)

        return [optim], [scheduler]


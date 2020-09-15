import torch.nn as nn
import pretrainedmodels
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import numpy as np


class LabelSmoothLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


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


class SeResNet(nn.Module):
    def __init__(self, layers, drop, num_classes):
        super().__init__()
        if layers == 50:
            model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')

        if layers == 101:
            model = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet')

        if layers == 152:
            model = pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained='imagenet')

        self.encoder = list(model.children())[:-2]

        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)

        if drop > 0:
            self.classifier = nn.Sequential(
                                            nn.Dropout(drop),
                                            nn.Linear(model.last_linear.in_features, num_classes))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(model.last_linear.in_features, num_classes)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.squeeze()


class SeResNet_PTL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        loss_class = eval(config.loss.class_name)
        self.criterion = loss_class()
        if config.model.params.layers == 50:
            model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')

        if config.model.params.layers == 101:
            model = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet')

        if config.model.params.layers == 152:
            model = pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained='imagenet')

        self.encoder = list(model.children())[:-2]

        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)

        if config.model.params.drop > 0:
            self.classifier = nn.Sequential(
                                            nn.Dropout(config.model.params.drop),
                                            nn.Linear(model.last_linear.in_features, config.model.params.num_classes))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(model.last_linear.in_features, config.model.params.num_classes)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_nb, optimizer_idx=0):
        x, y = batch

        y = y.long()

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

        y = y.long()

        logits = self(x)
        loss = self.criterion(logits, y)
        sm = torch.sigmoid(logits)
        pred = sm > 0.5
        # val_auc = roc_auc_score(y_true=y.cpu(), y_score=sm.cpu())
        val_acc = accuracy_score(y_true=y.cpu(), y_pred=pred[:,1].cpu())
        val_acc = torch.tensor(val_acc)
        log = {'val_loss': loss, 'val_acc': val_acc}
        return {'val_loss': loss, 'val_acc': val_acc, 'log': log}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self, classifier_only=True):
        if classifier_only:
            optim = torch.optim.Adam(self.classifier.parameters(), **self.config.optimizer.params)
        else:
            optim = torch.optim.Adam(self.parameters(), **self.config.optimizer.params)
        scheduler = MultiStepLR(optim, milestones=self.config.optimizer.scheduler.milestones,
                                gamma=self.config.optimizer.scheduler.gamma)
        # scheduler = ReduceLROnPlateau(optimizer=optim)
        return [optim], [scheduler]
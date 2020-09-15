import pickle


import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import numpy as np
import matplotlib.pyplot as plt
import os

from classifier.scripts.delong_test import delong_roc_test
from classifier.scripts.test import pr_curve_bootstrap, roc_curve_bootstrap

import statsmodels.api as sm

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def squared_error(x, y):
    return np.mean((x - y)**2)


if __name__ == '__main__':
    train_file = 'analysis/data_0_pred.csv'
    test1_file = 'analysis/data_1.csv'
    test2_file = 'analysis/data_2.csv'
    # load train data
    train_meta = pd.read_csv(train_file)
    gkf = GroupKFold(5)
    learned_models = list()
    val_proba_list = list()
    val_gt_list = list()
    data = train_meta.values[:, 2:4].astype(np.float32)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    for fold, (train_ind, val_ind) in enumerate(gkf.split(train_meta, train_meta.Prediction, train_meta.ID)):
        train_data = train_meta.iloc[train_ind].values
        val_data = train_meta.iloc[val_ind].values
        train_x, train_y = train_data[:,2:4], train_data[:, 4]
        train_x = np.asarray(train_x).astype(np.float32)
        train_x = (train_x - mean) / std
        train_y = np.asarray(train_y).astype(np.float32)
        # model = LogisticRegression(2)
        # _ = model.fit(train_x, train_y, 1e-1, 50)
        model = sm.Logit(train_y, train_x)
        result = model.fit()
        print(result.summary())
        learned_models.append(model)
        val_x, val_y = val_data[:, 2:4], val_data[:, 4]
        val_x = (val_x - mean) / std
        val_gt = val_data[:, 5]
        val_x = np.asarray(val_x).astype(np.float32)
        val_y = np.asarray(val_y).astype(np.float32)
        # loss = model.pdf(val_x)
        prob = model.pdf(val_x)
        val_proba_list += list(prob)
        val_gt_list += list(val_gt)
        # print(f'Fold #{fold} Validation Loss: ', loss)
        break
    # val_prob = np.asarray(val_proba_list)
    # val_gt = np.asarray(val_gt_list)
    # best_th = -1
    # max_f1 = 0.0
    # for th in range(1, 101):
    #     th = float(th / 100.0)
    #     y_pred = val_prob[:,1] > th
    #     f1 = f1_score(y_pred=y_pred, y_true=val_gt)
    #     if f1 > max_f1:
    #         max_f1 = f1
    #         best_th = th
    #
    # lr_folder = 'analysis/lr'
    # os.makedirs(lr_folder, exist_ok=True)
    # print(f'Best threshold is {best_th} and max f1 is {max_f1}')
    # print('Evaluating TS1')
    # test_meta = pd.read_csv(test1_file)
    # test_data = test_meta.values
    # test_x = np.asarray(test_data[:, 2:4]).astype(np.float32)
    # test_x = (test_x - mean) / std
    # test_y = np.asarray(test_data[:, 4]).astype(np.float32)
    # preds = np.zeros(shape=(len(test_y), 1)).astype(np.float)
    # for model in learned_models:
    #     out = model.pdf(test_x)[:,1]
    #     preds += out.reshape(preds.shape)
    #
    # preds /= len(learned_models)
    # # load deep wrist's prediction
    # with open('saved_data_ts1.pkl', 'rb') as f:
    #     gt_pa = pickle.load(f)
    #     gt_lat = pickle.load(f)
    #     predictions_pa = pickle.load(f)
    #     predictions_lat = pickle.load(f)
    #     ensemble_dict = pickle.load(f)
    #
    # ensemble_pred = list()
    # ensemble_gt = list()
    # for key in ensemble_dict:
    #     value = ensemble_dict[key]
    #     dw_preds = value['pred']
    #     gt = value['gt']
    #     pred = np.mean(dw_preds)
    #     ensemble_pred.append(pred)
    #     ensemble_gt.append(gt)
    # ensemble_gt = np.asarray(ensemble_gt)
    # ensemble_pred = np.asarray(ensemble_pred)
    #
    # p = delong_roc_test(ground_truth=ensemble_gt[:, 0], predictions_one=ensemble_pred, predictions_two=preds.reshape(preds.shape[0], ))
    # print(f'P value for TestSet 1: {np.exp2(p)}')
    #
    # acc = accuracy_score(y_pred=preds>= best_th, y_true=test_y)
    # auc = roc_auc_score(test_y, preds)
    # print(auc)
    # print(f'TS1 Acc: {acc}')
    # plt.rcParams.update({'font.size': 12})
    # auc_ens, CI_l_ens, CI_h_ens = roc_curve_bootstrap(test_y, preds,
    #                                                   os.path.join(lr_folder, 'testset1_AUROC.pdf'),
    #                                                   n_bootstrap=5000)
    # print(f'AUROC on TS1, {auc_ens}, CI {CI_l_ens} - {CI_h_ens}')
    # pr_ens, CI_l_pr_ens, CI_h_pr_ens = pr_curve_bootstrap(test_y, preds, os.path.join(lr_folder, 'testset1_AUPR.pdf'), model_name='LogisticReg')
    # print(f'AUPR on TS1, {pr_ens}, CI {CI_l_pr_ens} - {CI_h_pr_ens}')
    # print('Evaluating TS2')
    # test_meta = pd.read_csv(test2_file)
    # test_data = test_meta.values
    # test_x = np.asarray(test_data[:, 2:4]).astype(np.float32)
    # test_x = (test_x - mean) / std
    # test_y = np.asarray(test_data[:, 4]).astype(np.float32)
    # preds = np.zeros(shape=(len(test_y), 1)).astype(np.float)
    # for model in learned_models:
    #     out = model.pdf(test_x)[:, 1]
    #     preds += out.reshape(preds.shape)
    #
    # preds /= len(learned_models)
    #
    # with open('saved_data_ts2.pkl', 'rb') as f:
    #     gt_pa = pickle.load(f)
    #     gt_lat = pickle.load(f)
    #     predictions_pa = pickle.load(f)
    #     predictions_lat = pickle.load(f)
    #     ensemble_dict = pickle.load(f)
    #
    # ensemble_pred = list()
    # ensemble_gt = list()
    # for key in ensemble_dict:
    #     value = ensemble_dict[key]
    #     dw_preds = value['pred']
    #     gt = value['gt']
    #     pred = np.mean(dw_preds)
    #     ensemble_pred.append(pred)
    #     ensemble_gt.append(gt)
    # ensemble_gt = np.asarray(ensemble_gt)
    # ensemble_pred = np.asarray(ensemble_pred)
    #
    # p = delong_roc_test(ground_truth=ensemble_gt[:, 0], predictions_one=ensemble_pred, predictions_two=preds.reshape(preds.shape[0],))
    # print(f'P value for TestSet 2: {np.exp2(p)}')
    # acc = accuracy_score(y_pred=preds>=best_th, y_true=test_y)
    # print(f'TS2 Acc: {acc}')
    #
    # plt.rcParams.update({'font.size': 12})
    # auc_ens, CI_l_ens, CI_h_ens = roc_curve_bootstrap(test_y, preds,
    #                                                   os.path.join(lr_folder, 'testset2_AUROC.pdf'),
    #                                                   n_bootstrap=5000)
    # print(f'AUROC on TS2, {auc_ens}, CI {CI_l_ens} - {CI_h_ens}')
    # pr_ens, CI_l_pr_ens, CI_h_pr_ens = pr_curve_bootstrap(test_y, preds, os.path.join(lr_folder, 'testset2_AUPR.pdf'), model_name='LogisticReg')
    # print(f'AUPR on TS2, {pr_ens}, CI {CI_l_pr_ens} - {CI_h_pr_ens}')








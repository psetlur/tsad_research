import logging
import math
import itertools

import geomloss
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm


class EmbNormalizer:
    def __init__(self, mode="tpsd"):
        self.mode = mode
        self.emb_mean = None
        self.emb_std = None

    def __call__(self, emb_x, emb_y, emb_z):
        if self.mode == "tpsd":
            emb_all = torch.cat([emb_x, emb_y, emb_z])
            self.emb_mean = emb_all.mean(0)
            self.emb_std = torch.norm(emb_all - self.emb_mean) / math.sqrt(emb_all.size(0))
            emb_x = (emb_x - self.emb_mean) / self.emb_std
            emb_y = (emb_y - self.emb_mean) / self.emb_std
            emb_z = (emb_z - self.emb_mean) / self.emb_std
            return emb_x, emb_y, emb_z
        else:
            raise ValueError(self.mode)

    def normalize(self, emb):
        return (emb - self.emb_mean) / self.emb_std


def inject_platform(ts_row, level, start, length):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    start_a = int(len(ts_row) * start)
    length_a = int(len(ts_row) * length)
    ts_row[start_a: start_a + length_a] = level
    label[start_a: start_a + length_a] = 1
    return ts_row, label


def inject_mean(ts_row, level, start, length):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    start_a = int(len(ts_row) * start)
    length_a = int(len(ts_row) * length)
    ts_row[start_a: start_a + length_a] += float(level)
    label[start_a: start_a + length_a] = 1
    ts_row = ((ts_row - ts_row.min()) / (ts_row.max() - ts_row.min())) * 2 - 1
    return ts_row, label


def inject_spike(ts_row, level, start):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    start_a = int(len(ts_row) * start)
    ts_row[start_a] += level
    label[start_a] = 1
    return ts_row, label


def inject(anomaly_type, ts, config):
    if anomaly_type == 'platform':
        return inject_platform(ts, *config)
    elif anomaly_type == 'mean':
        return inject_mean(ts, *config)
    elif anomaly_type == 'spike':
        return inject_spike(ts, config[0], config[1])
    else:
        raise Exception('Unsupported anomaly_type.')


def train_classify_model(args, X_train, y_train):
    model = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 512),
    ).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(1000):
        out = model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def classify(model, X_valid):
    y_pred = torch.where(torch.sigmoid(model(X_valid).detach()) > 0.5, 1, 0).cpu().numpy()
    return y_pred


# def classify(X_train, y_train, X_test):
#     model = nn.Sequential(
#         nn.Linear(128, 128),
#         nn.ReLU(),
#         nn.Linear(128, 512),
#     ).to(args.device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#     for _ in range(1000):
#         out = model(X_train)
#         loss = criterion(out, y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     y_pred = torch.where(torch.sigmoid(model(X_test).detach()) > 0.5, 1, 0).cpu().numpy()
#     return y_pred


def hist_sample(cdf, bins):
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    val = np.random.uniform(bins[bin_idx - 1], bins[bin_idx])
    return val


# def black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, a_config):
#     ratio_anomaly = 0.1
#     fixed_level = 0.5
#     fixed_length = 0.3
#     train_levels = np.round(np.arange(-1.0, 1.1, 0.1), 1)
#     train_lengths = np.round(np.arange(0.2, 0.52, 0.02), 2)
#     valid_levels = np.round(np.arange(-1.0, 1.1, 0.1), 1)
#     valid_lengths = np.round(np.arange(0.2, 0.52, 0.02), 2)
#
#     if args.trail == 'inject_spike':
#         anomaly_types = ['platform', 'mean', 'spike']
#     elif args.trail == 'second_anomaly':
#         anomaly_types = ['platform', 'mean']
#     else:
#         raise Exception('Unsupported trail.')
#
#     with torch.no_grad():
#         z_train, x_train_np = [], []
#         for x_batch in train_dataloader:
#             c_x = model(x_batch.to(0)).detach()
#             z_train.append(c_x)
#             x_train_np.append(x_batch.numpy())
#         z_train = torch.cat(z_train, dim=0)
#         x_train_np = np.concatenate(x_train_np, axis=0).reshape(len(z_train), -1)
#
#         # z_valid = []
#         # for x_batch in val_dataloader:
#         #     c_x = model(x_batch.to(0)).detach()
#         #     z_valid.append(c_x)
#         # z_valid = torch.cat(z_valid, dim=0)
#
#         z_valid, x_valid_np = [], []
#         for x_batch in val_dataloader:
#             c_x = model(x_batch.to(0)).detach()
#             z_valid.append(c_x)
#             x_valid_np.append(x_batch.numpy())
#         z_valid = torch.cat(z_valid, dim=0)
#         x_valid_np = np.concatenate(x_valid_np, axis=0).reshape(len(z_valid), -1)
#
#         # z_test, y_test, t_test = [], [], []
#         # for x_batch, y_batch in test_dataloader:
#         #     c_x = model(x_batch.to(0)).detach()
#         #     z_test.append(c_x)
#         #     y_batch_t = np.zeros((x_batch.shape[0], x_batch.shape[2]))
#         #     for i, m in enumerate(y_batch.squeeze()):
#         #         m_start, m_length, _, m_type = m[-4:]
#         #         if m_type != -1:
#         #             y_batch_t[i, int(m_start):int(m_start) + int(m_length)] = 1
#         #     y_test.append(y_batch_t)
#         #     t_test.append(y_batch[:, 0, -1])
#         # z_test = torch.cat(z_test, dim=0)
#         # y_test = np.concatenate(y_test, axis=0)
#         # t_test = np.concatenate(t_test, axis=0)
#
#         emb = EmbNormalizer()
#         total_loss = dict()
#         f1score = dict()
#
#         train_inlier_index, train_outlier_index = train_test_split(range(len(x_train_np)),
#                                                                    train_size=1 - ratio_anomaly, random_state=0)
#         valid_inlier_index, valid_outlier_index = train_test_split(range(len(x_valid_np)),
#                                                                    train_size=1 - ratio_anomaly, random_state=0)
#
#         # test_index_0, test_index_1 = train_test_split(test_index, train_size=ratio_0 / (ratio_0 + ratio_1),
#         #                                               random_state=seed)
#
#         x_train_aug, labels_train = dict(), dict()
#         for anomaly_type in anomaly_types:
#             x_aug, labels = list(), list()
#             for i in train_outlier_index:
#                 # if np.random.random() > 0.5:
#                 #     xa, l = inject_platform(x, fixed_level_0, fixed_start_0, fixed_length_0)
#                 # else:
#                 #     xa, l = inject_platform(x, fixed_level_1, fixed_start_1, fixed_length_1)
#                 xa, l = inject(anomaly_type=anomaly_type, ts=x_train_np[i],
#                                config=[fixed_level, np.random.uniform(0, 0.5), fixed_length])
#                 x_aug.append(xa)
#                 labels.append(l)
#             x_train_aug[anomaly_type] = x_aug
#             labels_train[anomaly_type] = labels
#
#         x_valid_aug, labels_valid = dict(), dict()
#         for anomaly_type in anomaly_types:
#             x_aug, labels = list(), list()
#             for i in valid_outlier_index:
#                 xa, l = inject(anomaly_type=anomaly_type, ts=x_valid_np[i],
#                                config=[fixed_level, np.random.uniform(0, 0.5), fixed_length])
#                 x_aug.append(xa)
#                 labels.append(l)
#             x_valid_aug[anomaly_type] = x_aug
#             labels_valid[anomaly_type] = labels
#
#         z_aug_train = {anomaly_type: model(
#             torch.cat([torch.tensor(np.array(x_aug)).to(args.device)], dim=0).float().unsqueeze(1)).detach()
#                        for anomaly_type, x_aug in x_train_aug.items()}
#         z_aug_valid = {anomaly_type: model(
#             torch.cat([torch.tensor(np.array(x_aug)).to(args.device)], dim=0).float().unsqueeze(1)).detach()
#                        for anomaly_type, x_aug in x_valid_aug.items()}
#         z_train_t, z_valid_t, _ = emb(z_train[train_inlier_index].clone().squeeze(),
#                                       z_valid[valid_inlier_index].clone().squeeze(),
#                                       torch.cat([z_aug for z_aug in z_aug_train.values()], dim=0).squeeze())
#
#         def argument(x_np, configs, outlier_index, config_name):
#             x_configs_augs_dict, configs_labels_dict = dict(), dict()
#             for anomaly_type in anomaly_types:
#                 x_configs_augs, configs_labels = list(), list()
#                 for config in configs:
#                     x_augs, labels = list(), list()
#                     for i in outlier_index:
#                         x = x_np[i]
#                         if config_name == 'level':
#                             x_aug, label = inject(anomaly_type=anomaly_type, ts=x,
#                                                   config=[config, np.random.uniform(0, 0.5), fixed_length])
#                         else:  # config_name == 'length'
#                             if anomaly_type == 'spike':
#                                 break
#                             x_aug, label = inject(anomaly_type=anomaly_type, ts=x,
#                                                   config=[fixed_level, np.random.uniform(0, 0.5), config])
#                         x_augs.append(x_aug)
#                         labels.append(label)
#                     x_configs_augs.append(x_augs)
#                     configs_labels.append(labels)
#                 x_configs_augs_dict[anomaly_type] = x_configs_augs
#                 configs_labels_dict[anomaly_type] = configs_labels
#             return x_configs_augs_dict, configs_labels_dict
#
#         # train level aug
#         x_train_level_aug, train_level_labels = argument(x_np=x_train_np, configs=train_levels,
#                                                          outlier_index=train_outlier_index, config_name='level')
#         # train length aug
#         x_train_length_aug, train_length_labels = argument(x_np=x_train_np, configs=train_lengths,
#                                                            outlier_index=train_outlier_index, config_name='length')
#         # valid level aug
#         x_valid_level_aug, valid_level_labels = argument(x_np=x_valid_np, configs=valid_levels,
#                                                          outlier_index=valid_outlier_index, config_name='level')
#         # valid length aug
#         x_valid_length_aug, valid_length_labels = argument(x_np=x_valid_np, configs=valid_lengths,
#                                                            outlier_index=valid_outlier_index, config_name='length')
#
#         z_train_level_aug = {anomaly_type: [
#             model(torch.tensor(np.array(level_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
#             level_x_aug in x_train_level_aug[anomaly_type]] for anomaly_type in anomaly_types}
#         z_train_length_aug = {anomaly_type: [
#             model(torch.tensor(np.array(length_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
#             length_x_aug in x_train_length_aug[anomaly_type]] for anomaly_type in anomaly_types}
#         z_valid_level_aug = {anomaly_type: [
#             model(torch.tensor(np.array(level_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
#             level_x_aug in x_valid_level_aug[anomaly_type]] for anomaly_type in anomaly_types}
#         z_valid_length_aug = {anomaly_type: [
#             model(torch.tensor(np.array(length_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
#             length_x_aug in x_valid_length_aug[anomaly_type]] for anomaly_type in anomaly_types}
#
#         z_aug_train_t = {anomaly_type: [emb.normalize(z_aug) for z_aug in z_aug_train[anomaly_type]] for
#                          anomaly_type in anomaly_types}
#         z_aug_valid_t = {anomaly_type: [emb.normalize(z_aug) for z_aug in z_aug_valid[anomaly_type]] for
#                          anomaly_type in anomaly_types}
#
#         z_train_level_aug_t = {anomaly_type: [emb.normalize(z_aug) for z_aug in z_train_level_aug[anomaly_type]] for
#                                anomaly_type in anomaly_types}
#         z_train_length_aug_t = {anomaly_type: [emb.normalize(z_aug) for z_aug in z_train_length_aug[anomaly_type]]
#                                 for anomaly_type in anomaly_types}
#         z_valid_level_aug_t = {anomaly_type: [emb.normalize(z_aug) for z_aug in z_valid_level_aug[anomaly_type]] for
#                                anomaly_type in anomaly_types}
#         z_valid_length_aug_t = {anomaly_type: [emb.normalize(z_aug) for z_aug in z_valid_length_aug[anomaly_type]]
#                                 for anomaly_type in anomaly_types}
#
#         W_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
#
#         # loss = -W_loss(torch.cat([z_train_t, z_aug_t], dim=0), z_valid_t).item()
#         # total_loss.append(loss)
#         # z_test_t = emb.normalize(z_test)
#         # X = np.concatenate([z_train_t.numpy(), z_aug_t.numpy()], axis=0)
#         # y = np.concatenate([np.zeros((len(train_inlier_index), x_train_np.shape[1])), labels], axis=0)
#         # y_pred = classify(torch.tensor(X).float().to(args.device), torch.tensor(y).float().to(args.device),
#         #                   z_test_t.to(args.device))
#         # f1score.append(f1_score(y_test.reshape(-1), y_pred.reshape(-1)))
#
#         for train_anomaly in anomaly_types:
#             total_loss[train_anomaly] = dict()
#             f1score[train_anomaly] = dict()
#             other_train_anomaly = torch.cat(
#                 [torch.stack(z_aug_train_t[anomaly_type])
#                  for anomaly_type in anomaly_types if anomaly_type != train_anomaly], dim=0)
#             other_train_labels = np.concatenate(
#                 [np.vstack(labels_train[anomaly_type]) for anomaly_type in anomaly_types
#                  if anomaly_type != train_anomaly], axis=0)
#
#             total_loss[train_anomaly]['level'] = dict()
#             f1score[train_anomaly]['level'] = dict()
#             for i, train_level in enumerate(train_levels):
#                 total_loss[train_anomaly]['level'][train_level] = dict()
#                 f1score[train_anomaly]['level'][train_level] = dict()
#                 classify_model = None
#
#                 for valid_anomaly in anomaly_types:
#                     total_loss[train_anomaly]['level'][train_level][valid_anomaly] = dict()
#                     f1score[train_anomaly]['level'][train_level][valid_anomaly] = dict()
#                     other_valid_anomaly = torch.cat(
#                         [torch.stack(z_aug_valid_t[anomaly_type])
#                          for anomaly_type in anomaly_types if anomaly_type != valid_anomaly], dim=0)
#                     other_valid_labels = np.concatenate(
#                         [np.vstack(labels_valid[anomaly_type]) for anomaly_type in anomaly_types
#                          if anomaly_type != valid_anomaly], axis=0)
#
#                     total_loss[train_anomaly]['level'][train_level][valid_anomaly]['level'] = dict()
#                     f1score[train_anomaly]['level'][train_level][valid_anomaly]['level'] = dict()
#                     for j, valid_level in enumerate(valid_levels):
#                         loss = W_loss(
#                             torch.cat([z_train_level_aug_t[train_anomaly][i], other_train_anomaly], dim=0),
#                             torch.cat([z_valid_level_aug_t[valid_anomaly][j], other_valid_anomaly], dim=0)).item()
#                         total_loss[train_anomaly]['level'][train_level][valid_anomaly]['level'][valid_level] = loss
#                         if classify_model is None:
#                             X = torch.cat(
#                                 [z_train_t, z_train_level_aug_t[train_anomaly][i], other_train_anomaly], dim=0)
#                             y = torch.tensor(np.concatenate(
#                                 [np.zeros((len(train_inlier_index), x_train_np.shape[1])),
#                                  train_level_labels[train_anomaly][i], other_train_labels], axis=0)).to(args.device)
#                             classify_model = train_classify_model(args=args, X_train=X, y_train=y)
#                         y_pred = classify(model=classify_model, X_valid=torch.cat(
#                             [z_valid_level_aug_t[valid_anomaly][j], other_valid_anomaly], dim=0).to(args.device))
#                         f1 = f1_score(torch.tensor(
#                             np.concatenate([valid_level_labels[valid_anomaly][j], other_valid_labels],
#                                            axis=0)).reshape(-1), y_pred.reshape(-1))
#                         f1score[train_anomaly]['level'][train_level][valid_anomaly]['level'][valid_level] = f1
#                         print(f'train_anomaly.level: {train_anomaly}.{train_level}, '
#                               f'valid_anomaly.level: {valid_anomaly}.{valid_level}, wd: {loss}, f1score: {f1}')
#
#                     total_loss[train_anomaly]['level'][train_level][valid_anomaly]['length'] = dict()
#                     f1score[train_anomaly]['level'][train_level][valid_anomaly]['length'] = dict()
#                     for j, valid_length in enumerate(valid_lengths):
#                         loss = W_loss(
#                             torch.cat([z_train_level_aug_t[train_anomaly][i], other_train_anomaly], dim=0),
#                             torch.cat([z_valid_length_aug_t[valid_anomaly][j], other_valid_anomaly], dim=0)).item()
#                         total_loss[train_anomaly]['level'][train_level][valid_anomaly]['length'][valid_length] = loss
#                         if classify_model is None:
#                             X = torch.cat(
#                                 [z_train_t, z_train_level_aug_t[train_anomaly][i], other_train_anomaly], dim=0)
#                             y = torch.tensor(np.concatenate(
#                                 [np.zeros((len(train_inlier_index), x_train_np.shape[1])),
#                                  train_level_labels[train_anomaly][i], other_train_labels], axis=0)).to(args.device)
#                             classify_model = train_classify_model(args=args, X_train=X, y_train=y)
#                         y_pred = classify(model=classify_model, X_valid=torch.cat(
#                             [z_valid_length_aug_t[valid_anomaly][j], other_valid_anomaly], dim=0).to(args.device))
#                         f1 = f1_score(torch.tensor(
#                             np.concatenate([valid_length_labels[valid_anomaly][j], other_valid_labels],
#                                            axis=0)).reshape(-1), y_pred.reshape(-1))
#                         f1score[train_anomaly]['level'][train_level][valid_anomaly]['length'][valid_length] = f1
#                         print(f'train_anomaly.level: {train_anomaly}.{train_level}, '
#                               f'valid_anomaly.length: {valid_anomaly}.{valid_length}, wd: {loss}, f1score: {f1}')
#
#             total_loss[train_anomaly]['length'] = dict()
#             f1score[train_anomaly]['length'] = dict()
#             for i, train_length in enumerate(train_lengths):
#                 total_loss[train_anomaly]['length'][train_length] = dict()
#                 f1score[train_anomaly]['length'][train_length] = dict()
#                 classify_model = None
#
#                 for valid_anomaly in anomaly_types:
#                     total_loss[train_anomaly]['length'][train_length][valid_anomaly] = dict()
#                     f1score[train_anomaly]['length'][train_length][valid_anomaly] = dict()
#                     other_valid_anomaly = torch.cat(
#                         [torch.stack(z_aug_valid_t[anomaly_type])
#                          for anomaly_type in anomaly_types if anomaly_type != valid_anomaly], dim=0)
#                     other_valid_labels = np.concatenate(
#                         [np.vstack(labels_valid[anomaly_type]) for anomaly_type in anomaly_types
#                          if anomaly_type != valid_anomaly], axis=0)
#
#                     total_loss[train_anomaly]['length'][train_length][valid_anomaly]['level'] = dict()
#                     f1score[train_anomaly]['length'][train_length][valid_anomaly]['level'] = dict()
#                     for j, valid_level in enumerate(valid_levels):
#                         loss = W_loss(
#                             torch.cat([z_train_length_aug_t[train_anomaly][i], other_train_anomaly], dim=0),
#                             torch.cat([z_valid_level_aug_t[valid_anomaly][j], other_valid_anomaly], dim=0)).item()
#                         total_loss[train_anomaly]['length'][train_length][valid_anomaly]['level'][valid_level] = loss
#                         if classify_model is None:
#                             X = torch.cat(
#                                 [z_train_t, z_train_length_aug_t[train_anomaly][i], other_train_anomaly], dim=0)
#                             y = torch.tensor(np.concatenate(
#                                 [np.zeros((len(train_inlier_index), x_train_np.shape[1])),
#                                  train_length_labels[train_anomaly][i], other_train_labels], axis=0)).to(args.device)
#                             classify_model = train_classify_model(args=args, X_train=X, y_train=y)
#                         y_pred = classify(model=classify_model, X_valid=torch.cat(
#                             [z_valid_level_aug_t[valid_anomaly][j], other_valid_anomaly], dim=0).to(args.device))
#                         f1 = f1_score(torch.tensor(
#                             np.concatenate([valid_level_labels[valid_anomaly][j], other_valid_labels],
#                                            axis=0)).reshape(-1), y_pred.reshape(-1))
#                         f1score[train_anomaly]['length'][train_length][valid_anomaly]['level'][valid_level] = f1
#                         print(f'train_anomaly.length: {train_anomaly}.{train_length}, '
#                               f'valid_anomaly.level: {valid_anomaly}.{valid_level}, wd: {loss}, f1score: {f1}')
#
#                     total_loss[train_anomaly]['length'][train_length][valid_anomaly]['length'] = dict()
#                     f1score[train_anomaly]['length'][train_length][valid_anomaly]['length'] = dict()
#                     for j, valid_length in enumerate(valid_lengths):
#                         loss = W_loss(
#                             torch.cat([z_train_length_aug_t[train_anomaly][i], other_train_anomaly], dim=0),
#                             torch.cat([z_valid_length_aug_t[valid_anomaly][j], other_valid_anomaly], dim=0)).item()
#                         total_loss[train_anomaly]['length'][train_length][valid_anomaly]['length'][valid_length] =
#                         loss
#                         if classify_model is None:
#                             X = torch.cat(
#                                 [z_train_t, z_train_length_aug_t[train_anomaly][i], other_train_anomaly], dim=0)
#                             y = torch.tensor(np.concatenate(
#                                 [np.zeros((len(train_inlier_index), x_train_np.shape[1])),
#                                  train_length_labels[train_anomaly][i], other_train_labels], axis=0)
#                             ).to(args.device)
#                             classify_model = train_classify_model(args=args, X_train=X, y_train=y)
#                         y_pred = classify(model=classify_model, X_valid=torch.cat(
#                             [z_valid_length_aug_t[valid_anomaly][j], other_valid_anomaly], dim=0).to(args.device))
#                         f1 = f1_score(torch.tensor(
#                             np.concatenate([valid_length_labels[valid_anomaly][j], other_valid_labels],
#                                            axis=0)).reshape(-1), y_pred.reshape(-1))
#                         f1score[train_anomaly]['length'][train_length][valid_anomaly]['length'][valid_length] = f1
#                         print(f'train_anomaly.length: {train_anomaly}.{train_length}, '
#                               f'valid_anomaly.length: {valid_anomaly}.{valid_length}, wd: {loss}, f1score: {f1}')
#
#         # for anomaly_type in anomaly_types:
#         #     # with test
#         #     visualize(train_inlier=z_train, valid_inlier=z_valid, train_augs=z_train_level_aug[anomaly_type],
#         #               valid_augs=z_valid_level_aug[anomaly_type], train_configs=train_levels,
#         #               valid_configs=valid_levels, config_name='level', fixed_config=f'fixed_length{fixed_length}',
#         #               trail=args.trail, test=True, anomaly_type=anomaly_type)
#         #     visualize(train_inlier=z_train, valid_inlier=z_valid, train_augs=z_train_length_aug[anomaly_type],
#         #               valid_augs=z_valid_length_aug[anomaly_type], train_configs=train_lengths,
#         #               valid_configs=valid_lengths, config_name='length', fixed_config=f'fixed_level{fixed_level}',
#         #               trail=args.trail, test=True, anomaly_type=anomaly_type)
#         #
#         #     # without test
#         #     visualize(train_inlier=z_train, valid_inlier=z_valid, train_augs=z_train_level_aug[anomaly_type],
#         #               valid_augs=None, train_configs=train_levels, valid_configs=None, config_name='level',
#         #               fixed_config=f'fixed_length{fixed_length}', trail=args.trail, test=False,
#         #               anomaly_type=anomaly_type)
#         #     visualize(train_inlier=z_train, valid_inlier=z_valid, train_augs=z_train_length_aug[anomaly_type],
#         #               valid_augs=None, train_configs=train_lengths, valid_configs=None, config_name='length',
#         #               fixed_config=f'fixed_level{fixed_level}', trail=args.trail, test=False,
#         # anomaly_type = anomaly_type)
#
#         # outlier = {'train': {'level': z_train_level_aug_t, 'length': z_train_length_aug_t},
#         #            'valid': {'level': z_valid_level_aug_t, 'length': z_valid_length_aug_t}}
#         # inlier = {'train': z_train, 'valid': z_valid}
#         # configs = {'level': train_levels, 'length': train_lengths}
#         # visualize_multiple(inlier=inlier, outlier=outlier, configs=configs, trail=args.trail,
#         #                    anomaly_types=anomaly_types)
#
#     return total_loss, f1score


def black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, a_config):
    ratio_anomaly = 0.1
    min_level = -1.0
    min_length = 0.2
    max_level = 2.0
    max_length = 0.6
    level_step = 2.0
    length_step = 0.3
    fixed_level = 0.5
    fixed_length = 0.3

    anomaly_types = ['platform', 'mean', 'spike']

    with torch.no_grad():
        z_train, x_train_np = [], []
        for x_batch in train_dataloader:
            c_x = model(x_batch.to(0)).detach()
            z_train.append(c_x)
            x_train_np.append(x_batch.numpy())
        z_train = torch.cat(z_train, dim=0)
        x_train_np = np.concatenate(x_train_np, axis=0).reshape(len(z_train), -1)

        z_valid, x_valid_np = [], []
        for x_batch in val_dataloader:
            c_x = model(x_batch.to(0)).detach()
            z_valid.append(c_x)
            x_valid_np.append(x_batch.numpy())
        z_valid = torch.cat(z_valid, dim=0)
        x_valid_np = np.concatenate(x_valid_np, axis=0).reshape(len(z_valid), -1)

        train_inlier_index, train_outlier_index = train_test_split(range(len(x_train_np)),
                                                                   train_size=1 - ratio_anomaly, random_state=0)
        valid_inlier_index, valid_outlier_index = train_test_split(range(len(x_valid_np)),
                                                                   train_size=1 - ratio_anomaly, random_state=0)
        x_aug_list = list()
        for anomaly_type in anomaly_types:
            x_augs = list()
            for i in train_outlier_index:
                x_aug, _ = inject(anomaly_type=anomaly_type, ts=x_train_np[i],
                                  config=[fixed_level, np.random.uniform(0, 0.5), fixed_length])
                x_augs.append(x_aug)
            x_aug_list.append(np.array(x_augs))

        emb = EmbNormalizer()
        z_aug = model(torch.tensor(np.concatenate(x_aug_list, axis=0)).float().unsqueeze(1).to(0)).detach()
        z_train_t, z_valid_t, _ = emb(z_train[train_inlier_index].clone().squeeze(),
                                      z_valid[valid_inlier_index].clone().squeeze(), z_aug)

        def argument(x_np, outlier_index):
            z_augs_dict, labels_dict = dict(), dict()
            x_augs, labels = dict(), dict()
            for anomaly_type in anomaly_types:
                x_augs[anomaly_type] = dict()
                labels[anomaly_type] = dict()

                def arg(level, length):
                    if anomaly_type != 'spike':
                        x_augs[anomaly_type][(round(level, 1), round(length, 1))] = list()
                        labels[anomaly_type][(round(level, 1), round(length, 1))] = list()
                    else:
                        x_augs[anomaly_type][(round(level, 1))] = list()
                        labels[anomaly_type][(round(level, 1))] = list()
                    for i in outlier_index:
                        x_aug, label = inject(anomaly_type=anomaly_type, ts=x_np[i],
                                              config=[level, np.random.uniform(0, 0.5), length])
                        if anomaly_type != 'spike':
                            x_augs[anomaly_type][(round(level, 1), round(length, 1))].append(np.array(x_aug))
                            labels[anomaly_type][(round(level, 1), round(length, 1))].append(np.array(label))
                        else:
                            x_augs[anomaly_type][(round(level, 1))].append(np.array(x_aug))
                            labels[anomaly_type][(round(level, 1))].append(np.array(label))

                for level in np.arange(min_level, max_level, level_step):
                    for length in np.arange(min_length, max_length, length_step):
                        arg(level=level, length=length)
                        if anomaly_type != 'spike':
                            x_augs[anomaly_type][(round(level, 1), round(length, 1))] = np.array(
                                x_augs[anomaly_type][(round(level, 1), round(length, 1))])
                            x_augs[anomaly_type][(round(level, 1), round(length, 1))] = torch.tensor(
                                x_augs[anomaly_type][(round(level, 1), round(length, 1))]).float().unsqueeze(1).to(
                                args.device)
                            x_augs[anomaly_type][(round(level, 1), round(length, 1))] = model(
                                x_augs[anomaly_type][(round(level, 1), round(length, 1))]).detach()
                        else:
                            x_augs[anomaly_type][(round(level, 1))] = np.array(
                                x_augs[anomaly_type][(round(level, 1))])
                            x_augs[anomaly_type][(round(level, 1))] = torch.tensor(
                                x_augs[anomaly_type][(round(level, 1))]).float().unsqueeze(1).to(args.device)
                            x_augs[anomaly_type][(round(level, 1))] = model(
                                x_augs[anomaly_type][(round(level, 1))]).detach()

            combinations = list(itertools.product(*[x_augs[anomaly_type] for anomaly_type in anomaly_types]))
            for combination in combinations:
                data = (list(), list())
                for anomaly_type, ac in zip(anomaly_types, combination):
                    data[0].append(x_augs[anomaly_type][ac])
                    data[1].append(labels[anomaly_type][ac])
                z_augs_dict[combination] = torch.cat(data[0], dim=0)
                labels_dict[combination] = np.concatenate(data[1], axis=0)
                # print(combination)
            return z_augs_dict, labels_dict

        z_train_aug, train_labels = argument(x_np=x_train_np, outlier_index=train_outlier_index)
        z_valid_aug, valid_labels = argument(x_np=x_valid_np, outlier_index=valid_outlier_index)

        z_train_aug_t = {config: emb.normalize(z_aug) for config, z_aug in z_train_aug.items()}
        z_valid_aug_t = {config: emb.normalize(z_aug) for config, z_aug in z_valid_aug.items()}

        total_loss = dict()
        f1score = dict()
        # W_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
        # for train_config, z_train_config_aug_t in z_train_aug_t.items():
        #     total_loss[train_config] = dict()
        #     f1score[train_config] = dict()
        #
        #     classify_model = None
        #     for valid_config, z_valid_config_aug_t in z_valid_aug_t.items():
        #         total_loss[train_config][valid_config] = W_loss(z_train_config_aug_t, z_valid_config_aug_t).item()
        #         if classify_model is None:
        #             X = torch.cat([z_train_t, z_train_config_aug_t], dim=0)
        #             y = torch.tensor(np.concatenate([np.zeros((len(train_inlier_index), x_train_np.shape[1])),
        #                                              train_labels[train_config]], axis=0)).to(args.device)
        #             classify_model = train_classify_model(args=args, X_train=X, y_train=y)
        #
        #         y_pred = classify(model=classify_model, X_valid=z_valid_config_aug_t)
        #         f1score[train_config][valid_config] = f1_score(torch.tensor(valid_labels[valid_config]).
        #                                                        reshape(-1), y_pred.reshape(-1))
        #         print(f'train: {train_config}, valid: {valid_config}, '
        #               f'wd: {total_loss[train_config][valid_config]}, '
        #               f'f1score: {f1score[train_config][valid_config]}')

        inlier = {'train': z_train, 'valid': z_valid}
        outlier = {'train': z_train_aug_t, 'valid': z_valid_aug_t}
        combinations = list(z_train_aug_t.keys())
        visualize_multiple(inlier=inlier, outlier=outlier, combinations=combinations, trail=args.trail)

    return total_loss, f1score


# COLORS = {'level': {'train': {'platform': 'Greys', 'mean': 'Oranges'},
#                     'valid': {'platform': 'Reds', 'mean': 'Purples'}},
#           'length': {'train': {'platform': 'viridis', 'mean': 'plasma'},
#                      'valid': {'platform': 'BuPu', 'mean': 'GnBu'}}}

from matplotlib.colors import LinearSegmentedColormap


def create_dark_green_cmap(name="DarkGreen"):
    cdict = {
        'red': [(0.0, 0.8, 0.8), (1.0, 0.0, 0.0)],
        'green': [(0.0, 1.0, 1.0), (1.0, 0.2, 0.2)],
        'blue': [(0.0, 0.8, 0.8), (1.0, 0.0, 0.0)],
    }
    return LinearSegmentedColormap(name, cdict)


def create_brown_cmap(name="Brown"):
    cdict = {
        'red': [(0.0, 1.0, 1.0), (1.0, 0.5, 0.5)],
        'green': [(0.0, 0.8, 0.8), (1.0, 0.2, 0.2)],
        'blue': [(0.0, 0.6, 0.6), (1.0, 0.0, 0.0)],
    }
    return LinearSegmentedColormap(name, cdict)


dark_green_cmap = create_dark_green_cmap()
brown_cmap = create_brown_cmap()

COLORS = {
    'level': {
        'train': {
            'platform': 'Blues',
            'mean': 'Reds'
        },
        'valid': {
            'platform': 'Purples',
            'mean': 'Oranges'
        }
    },
    'length': {
        'train': {
            'platform': 'Greens',
            'mean': 'Greys'
        },
        'valid': {
            'platform': dark_green_cmap,
            'mean': brown_cmap
        }
    }
}


def visualize_multiple(inlier, outlier, combinations, trail):
    inlier_data = torch.cat([inlier['train'], inlier['valid']], dim=0)
    outlier_data = torch.cat([torch.cat([outlier['train'][combination] for combination in combinations], dim=0),
                              torch.cat([outlier['valid'][combination] for combination in combinations], dim=0)], dim=0)
    all_data = torch.cat([inlier_data, outlier_data], dim=0).to('cpu').numpy()
    inlier_size = len(inlier_data)
    xt = TSNE(n_components=2, random_state=42).fit_transform(all_data)
    plt.figure(figsize=(12, 10))

    # train and test inliers
    plt.scatter(xt[:len(inlier['train']), 0], xt[:len(inlier['train']), 1], c='b', alpha=0.5, label='Train inliers')
    plt.scatter(xt[len(inlier['train']): len(inlier['train']) + len(inlier['valid']), 0],
                xt[len(inlier['train']): len(inlier['train']) + len(inlier['valid']), 1],
                c='g', alpha=0.5, label='Test inliers')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Train inliers', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Test inliers', markerfacecolor='g', markersize=10)
    ]

    r = np.arange(len(combinations))
    normalized_values = (r - np.min(r)) / (np.max(r) - np.min(r))
    normalized_values = normalized_values * (1 - 0.1) + 0.1
    # train outliers
    cmap = cm.get_cmap('Greys')
    colors = [cmap(val) for val in normalized_values]
    start_idx = inlier_size
    for i, combination in enumerate(combinations):
        end_idx = start_idx + len(outlier['train'][combination])
        plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1], c=[colors[i]] * (end_idx - start_idx),
                    alpha=0.5)
        start_idx = end_idx
    for i, combination in enumerate(combinations):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Train {combination}',
                                      markerfacecolor=colors[i], markersize=10))

    # valid outliers
    cmap = cm.get_cmap('Reds')
    colors = [cmap(val) for val in normalized_values]
    for i, combination in enumerate(combinations):
        end_idx = start_idx + len(outlier['valid'][combination])
        plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1], c=[colors[i]] * (end_idx - start_idx),
                    alpha=0.5)
        start_idx = end_idx
    for i, combination in enumerate(combinations):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Test {combination}',
                                      markerfacecolor=colors[i], markersize=10))

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization of Embeddings')
    plt.tight_layout()
    plt.savefig(f'logs/training/{trail}/visualization.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    num_columns = 3  # Number of columns for the legend
    plt.figure(figsize=(12, len(legend_elements) * 0.2 / num_columns))
    plt.axis('off')
    plt.legend(handles=legend_elements, loc='center', ncol=num_columns, fontsize=8)
    plt.savefig(f'logs/training/{trail}/legend.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

# def visualize_multiple(inlier, outlier, configs, trail, anomaly_types):
#     inlier_data = torch.cat([inlier['train'], inlier['valid']], dim=0)
#     outlier_data = torch.cat([torch.cat([torch.cat(outlier['train']['level'][anomaly_type], dim=0)
#                                          for anomaly_type in anomaly_types], dim=0),
#                               torch.cat([torch.cat(outlier['valid']['level'][anomaly_type], dim=0)
#                                          for anomaly_type in anomaly_types], dim=0),
#                               torch.cat([torch.cat(outlier['train']['length'][anomaly_type], dim=0)
#                                          for anomaly_type in anomaly_types], dim=0),
#                               torch.cat([torch.cat(outlier['valid']['length'][anomaly_type], dim=0)
#                                          for anomaly_type in anomaly_types], dim=0)], dim=0)
#     all_data = torch.cat([inlier_data, outlier_data], dim=0).to('cpu').numpy()
#     inlier_size = len(inlier_data)
#     xt = TSNE(n_components=2, random_state=42).fit_transform(all_data)
#     plt.figure(figsize=(12, 10))
#
#     # train and test inliers
#     plt.scatter(xt[:len(inlier['train']), 0], xt[:len(inlier['train']), 1], c='b', alpha=0.5, label='Train inliers')
#     plt.scatter(xt[len(inlier['train']): len(inlier['train']) + len(inlier['valid']), 0],
#                 xt[len(inlier['train']): len(inlier['train']) + len(inlier['valid']), 1],
#                 c='g', alpha=0.5, label='Test inliers')
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='Train inliers', markerfacecolor='b', markersize=10),
#         Line2D([0], [0], marker='o', color='w', label='Test inliers', markerfacecolor='g', markersize=10)
#     ]
#
#     _start_idx = inlier_size
#
#     def plot_outlier(start_idx, config_name, process):
#         for anomaly_type in anomaly_types:
#             cmap = cm.get_cmap(COLORS[config_name][process][anomaly_type])
#             config_values = np.array(configs[config_name])
#             normalized = (config_values - np.min(config_values)) / (np.max(config_values) - np.min(config_values))
#             normalized = normalized * (1 - 0.1) + 0.1
#             colors = [cmap(val) for val in normalized]
#             for i, value in enumerate(config_values):
#                 end_idx = start_idx + len(outlier[process][config_name][anomaly_type][i])
#                 plt.scatter(xt[start_idx: end_idx, 0], xt[start_idx: end_idx, 1], c=[colors[i]] * (end_idx -
#                 start_idx),
#                             alpha=0.5)
#                 start_idx = end_idx
#                 legend_elements.append(Line2D([0], [0], marker='o', color='w',
#                                               label=f'{process} outliers ({anomaly_type}_{config_name}={value})',
#                                               markerfacecolor=colors[i], markersize=10))
#         return start_idx
#
#     _start_idx = plot_outlier(start_idx=_start_idx, config_name='level', process='train')
#     _start_idx = plot_outlier(start_idx=_start_idx, config_name='level', process='valid')
#     _start_idx = plot_outlier(start_idx=_start_idx, config_name='length', process='train')
#     _start_idx = plot_outlier(start_idx=_start_idx, config_name='length', process='valid')
#
#     plt.xlabel('t-SNE 1')
#     plt.ylabel('t-SNE 2')
#     plt.title('t-SNE Visualization of Embeddings')
#     plt.tight_layout()
#     plt.savefig(f'logs/training/{trail}/visualization.pdf', bbox_inches='tight')
#     plt.show()
#     plt.close()
#
#     num_columns = 3  # Number of columns for the legend
#     plt.figure(figsize=(12, len(legend_elements) * 0.2 / num_columns))
#     plt.axis('off')
#     plt.legend(handles=legend_elements, loc='center', ncol=num_columns, fontsize=8)
#     plt.savefig(f'logs/training/{trail}/legend.pdf', bbox_inches='tight')
#     plt.show()
#     plt.close()

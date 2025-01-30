import numpy as np
import math

import torch
from torch import nn
import geomloss

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import logging
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import random


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


def train_classify_model(X_train, y_train):
    model = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 512),
    ).to('cuda:0')
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
#     ).to('cuda:0')
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


# def black_box_function(model, train_dataloader, val_dataloader, test_dataloader, a_config):
#     n_trials = 1
#     device = next(model.parameters()).device
#
#     ratio_anomaly = 0.1
#     levels = np.round(np.arange(-1.0, 1.1, 0.1), 1)
#     lengths = np.round(np.arange(0.2, 0.52, 0.02), 2)
#     fixed_level = 0.5
#     fixed_length = 0.3
#     fixed_start = 0.2
#
#     train_levels = np.round(np.arange(-1.0, 1.2, 0.2), 1)
#     # train_lengths = np.round(np.arange(0.2, 0.54, 0.04), 2)
#
#     with torch.no_grad():
#         # get embeddings for train, validation, and test data
#         def get_embeddings(dataloader):
#             z_data, x_data_np = [], []
#             for x_batch in dataloader:
#                 c_x = model(x_batch.to(device)).detach().cpu()
#                 z_data.append(c_x)
#                 x_data_np.append(x_batch.numpy())
#             z_data = torch.cat(z_data, dim=0)
#             x_data_np = np.concatenate(x_data_np, axis=0).reshape(len(z_data), -1)
#             return z_data, x_data_np
#
#         z_train, x_train_np = get_embeddings(train_dataloader)
#         z_valid, x_valid_np = get_embeddings(val_dataloader)
#
#         z_test, y_test, t_test = [], [], []
#         for x_batch, y_batch in test_dataloader:
#             c_x = model(x_batch.to(device)).detach().cpu()
#             z_test.append(c_x)
#             y_batch_t = np.zeros((x_batch.shape[0], x_batch.shape[2]))
#             for i, m in enumerate(y_batch.squeeze()):
#                 m_start, m_length, _, m_type = m[-4:]
#                 if m_type != -1:
#                     y_batch_t[i, int(m_start):int(m_start) + int(m_length)] = 1
#             y_test.append(y_batch_t)
#             t_test.append(y_batch[:, 0, -1])
#         z_test = torch.cat(z_test, dim=0)
#         y_test = np.concatenate(y_test, axis=0)
#         t_test = np.concatenate(t_test, axis=0)
#
#         emb = EmbNormalizer()
#         total_loss = []
#         fscore = []
#
#         for train_level in train_levels:
#             train_index, ttest_index = train_test_split(range(len(x_train_np)), train_size=1 - ratio_anomaly,
#                                                         random_state=0)
#             valid_index, vtest_index = train_test_split(range(len(x_valid_np)), train_size=1 - ratio_anomaly,
#                                                         random_state=0)
#
#             # inject test anomalies for all test levels
#             x_aug_level_list, labels_level_list = [], []
#             for level in levels:
#                 x_aug, labels = [], []
#                 for i in vtest_index:
#                     x = x_valid_np[i]
#                     xa, l = inject_platform(x, level, fixed_start, random.choice(lengths))
#                     x_aug.append(xa)
#                     labels.append(l)
#                 x_aug_level_list.append(x_aug)
#                 labels_level_list.append(labels)
#
#             x_aug_length_list, labels_length_list = [], []
#             for length in lengths:
#                 x_aug, labels = [], []
#                 for i in vtest_index:
#                     x = x_valid_np[i]
#                     xa, l = inject_platform(x, train_level, fixed_start, length)
#                     x_aug.append(xa)
#                     labels.append(l)
#                 x_aug_length_list.append(x_aug)
#                 labels_length_list.append(labels)
#
#             train_x_aug_level_list, train_labels_level_list = [], []
#             x_aug, labels = [], []
#             for i in ttest_index:
#                 x = x_train_np[i]
#                 xa, l = inject_platform(x, train_level, fixed_start, fixed_length)
#                 x_aug.append(xa)
#                 labels.append(l)
#             train_x_aug_level_list.append(x_aug)
#             train_labels_level_list.append(labels)
#
#             # getting embddings for training and testing anomalies
#             z_aug_level_list = [
#                 model(torch.tensor(np.array(x_aug_level_list)).float().unsqueeze(1).to(device)).detach().cpu()
#                 for x_aug_level in x_aug_level_list]
#             # z_aug_length_list = [model(torch.tensor(np.array(x_aug_length)).float().unsqueeze(1).to(device)).detach().cpu()
#             #       for x_aug_length in x_aug_length_list]
#
#             z_train_aug_level_list = [
#                 model(torch.tensor(np.array(x_aug_level)).float().unsqueeze(1).to(device)).detach().cpu() \
#                 for x_aug_level in train_x_aug_level_list]
#             # z_aug_level_list = [
#             #     model(torch.tensor(np.array(x_aug_level)).float().unsqueeze(1).to(device)).detach().cpu()
#             #     for x_aug_level in x_aug_level_list
#             # ]
#             # z_aug_length_list = [
#             #     model(torch.tensor(np.array(x_aug_length)).float().unsqueeze(1).to(device)).detach().cpu()
#             #     for x_aug_length in x_aug_length_list
#             # ]
#
#             z_aug = model(torch.tensor(np.array(x_aug)).float().unsqueeze(1).to(device)).detach().cpu()
#
#             # Normalize embeddings
#             z_train_t, z_aug_t, z_valid_t = emb(
#                 z_train[train_index].clone().squeeze(),
#                 z_aug.clone().squeeze(),
#                 z_valid[valid_index].clone().squeeze())
#
#             z_aug_t_level_list = [emb.normalize(z_aug_level) for z_aug_level in z_aug_level_list]
#             z_train_aug_t_level_list = [emb.normalize(z_aug_level) for z_aug_level in z_train_aug_level_list]
#
#             # computing WD loss
#             W_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
#             loss = -W_loss(torch.cat(
#                 [z_train_t, torch.cat(z_train_aug_t_level_list, dim=0),
#                  torch.ca(z_valid_t, torch.cat(z_aug_t_level_list, dim=0))]
#             ))
#             total_loss.append(loss)
#
#             # Compute F1 score
#             X = np.concatenate([z_train_t.numpy(), torch.cat(z_train_aug_t_level_list, dim=0).numpy()], axis=0)
#             y = np.concatenate(
#                 [
#                     np.zeros((len(train_index), x_train_np.shape[1])),
#                     np.concatenate(labels_level_list + labels_length_list, axis=0),
#                 ],
#                 axis=0,
#             )
#             y_pred = classify(
#                 torch.tensor(X).float().to(device),
#                 torch.tensor(y).float().to(device),
#                 torch.tensor(
#                     np.concatenate([
#                         z_valid_t,
#                         np.concatenate(z_aug_t_level_list, axis=0)
#                     ], axis=0)
#                 ).float().to(device),
#             )
#             y_valid = np.concatenate(
#                 [np.zeros((len(valid_index), x_valid_np.shape[1])), np.concatenate(labels_level_list, axis=0)], axis=0
#             )
#             fscore.append(f1_score(y_valid.reshape(-1), y_pred.reshape(-1)))
#
#             visualize_fixed_grid(z_train, z_valid, z_train_aug, z_aug_level_list, train_levels, levels,
#                                  'level', f'length{fixed_length}', 1)
#             visualize_fixed_grid(z_train, z_valid, z_train_aug, z_aug_level_list, train_levels, levels,
#                                  'level', f'length{fixed_length}', -1)
#
#     return total_loss, fscore

def black_box_function(model, train_dataloader, val_dataloader, test_dataloader, a_config):
    n_trials = 1
    # ratio_0, ratio_1 = a_config['ratio_0'], a_config['ratio_1']
    # ratio_anomaly = a_config['ratio_anomaly']
    # fixed_level = a_config['fixed_level']
    # fixed_length = a_config['fixed_length']
    # fixed_start = a_config['fixed_start']
    ratio_anomaly = 0.1
    fixed_level = 0.5
    fixed_length = 0.3
    train_levels = np.round(np.arange(-1.0, 1.1, 0.1), 1)
    train_lengths = np.round(np.arange(0.2, 0.52, 0.02), 2)
    valid_levels = np.round(np.arange(-1.0, 1.1, 0.1), 1)
    valid_lengths = np.round(np.arange(0.2, 0.52, 0.02), 2)

    with torch.no_grad():
        z_train, x_train_np = [], []
        for x_batch in train_dataloader:
            c_x = model(x_batch.to(0)).detach().cpu()
            z_train.append(c_x)
            x_train_np.append(x_batch.numpy())
        z_train = torch.cat(z_train, dim=0)
        x_train_np = np.concatenate(x_train_np, axis=0).reshape(len(z_train), -1)

        # z_valid = []
        # for x_batch in val_dataloader:
        #     c_x = model(x_batch.to(0)).detach().cpu()
        #     z_valid.append(c_x)
        # z_valid = torch.cat(z_valid, dim=0)

        z_valid, x_valid_np = [], []
        for x_batch in val_dataloader:
            c_x = model(x_batch.to(0)).detach().cpu()
            z_valid.append(c_x)
            x_valid_np.append(x_batch.numpy())
        z_valid = torch.cat(z_valid, dim=0)
        x_valid_np = np.concatenate(x_valid_np, axis=0).reshape(len(z_valid), -1)

        # z_test, y_test, t_test = [], [], []
        # for x_batch, y_batch in test_dataloader:
        #     c_x = model(x_batch.to(0)).detach().cpu()
        #     z_test.append(c_x)
        #     y_batch_t = np.zeros((x_batch.shape[0], x_batch.shape[2]))
        #     for i, m in enumerate(y_batch.squeeze()):
        #         m_start, m_length, _, m_type = m[-4:]
        #         if m_type != -1:
        #             y_batch_t[i, int(m_start):int(m_start) + int(m_length)] = 1
        #     y_test.append(y_batch_t)
        #     t_test.append(y_batch[:, 0, -1])
        # z_test = torch.cat(z_test, dim=0)
        # y_test = np.concatenate(y_test, axis=0)
        # t_test = np.concatenate(t_test, axis=0)

        emb = EmbNormalizer()
        # total_loss = []
        # f1score = []
        total_loss = dict()
        f1score = dict()
        for seed in range(n_trials):
            train_normal_index, train_abnormal_index = train_test_split(range(len(x_train_np)),
                                                                        train_size=1 - ratio_anomaly, random_state=seed)
            valid_normal_index, valid_abnormal_index = train_test_split(range(len(x_valid_np)),
                                                                        train_size=1 - ratio_anomaly, random_state=seed)

            # test_index_0, test_index_1 = train_test_split(test_index, train_size=ratio_0/(ratio_0+ratio_1), random_state=seed)

            def inject(x_np, configs, abnormal_index, config_name):
                x_aug_configs, labels_configs = [], []
                for config in configs:
                    x_aug, labels = [], []
                    for i in abnormal_index:
                        x = x_np[i]
                        if config_name == 'level':
                            xa, l = inject_platform(x, config, np.random.uniform(0, 0.5), fixed_length)
                        elif config_name == 'length':
                            xa, l = inject_platform(x, fixed_level, np.random.uniform(0, 0.5), config)
                        else:
                            raise Exception('Unsupported config')
                        x_aug.append(xa)
                        labels.append(l)
                    x_aug_configs.append(x_aug)
                    labels_configs.append(labels)
                return x_aug_configs, labels_configs

            # train aug level
            train_x_aug_level, train_labels_level = inject(x_np=x_train_np, configs=train_levels,
                                                           abnormal_index=train_abnormal_index, config_name='level')
            # train aug length
            train_x_aug_length, train_labels_length = inject(x_np=x_train_np, configs=train_lengths,
                                                             abnormal_index=train_abnormal_index, config_name='length')
            # valid aug level
            valid_x_aug_level, valid_labels_level = inject(x_np=x_valid_np, configs=valid_levels,
                                                           abnormal_index=valid_abnormal_index, config_name='level')
            # valid aug length
            valid_x_aug_length, valid_labels_length = inject(x_np=x_valid_np, configs=valid_lengths,
                                                             abnormal_index=valid_abnormal_index, config_name='length')

            x_aug, labels = [], []
            for i in train_abnormal_index:
                x = x_train_np[i]
                # if np.random.random() > 0.5:
                #     xa, l = inject_platform(x, fixed_level_0, fixed_start_0, fixed_length_0)
                # else:
                #     xa, l = inject_platform(x, fixed_level_1, fixed_start_1, fixed_length_1)
                xa, l = inject_platform(x, fixed_level, np.random.uniform(0, 0.5), fixed_length)
                x_aug.append(xa)
                labels.append(l)

            z_train_aug_level = [model(torch.tensor(np.array(x_aug_level)).float().unsqueeze(1).to(0)).detach().cpu()
                                 for x_aug_level in train_x_aug_level]
            z_train_aug_length = [model(torch.tensor(np.array(x_aug_length)).float().unsqueeze(1).to(0)).detach().cpu()
                                  for x_aug_length in train_x_aug_length]
            z_valid_aug_level = [model(torch.tensor(np.array(x_aug_level)).float().unsqueeze(1).to(0)).detach().cpu()
                                 for x_aug_level in valid_x_aug_level]
            z_valid_aug_length = [model(torch.tensor(np.array(x_aug_length)).float().unsqueeze(1).to(0)).detach().cpu()
                                  for x_aug_length in valid_x_aug_length]

            z_aug = model(torch.tensor(np.array(x_aug)).float().unsqueeze(1).to(0)).detach().cpu()
            z_train_t, z_valid_t, z_aug_t = emb(z_train[train_normal_index].clone().squeeze(),
                                                z_valid[valid_normal_index].clone().squeeze(),
                                                z_aug.clone().squeeze())
            z_train_aug_t_level = [emb.normalize(z_aug) for z_aug in z_train_aug_level]
            z_train_aug_t_length = [emb.normalize(z_aug) for z_aug in z_train_aug_length]
            z_valid_aug_t_level = [emb.normalize(z_aug) for z_aug in z_valid_aug_level]
            z_valid_aug_t_length = [emb.normalize(z_aug) for z_aug in z_valid_aug_length]

            W_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
            # loss = -W_loss(torch.cat([z_train_t, z_aug_t], dim=0), z_valid_t).item()
            # loss = -W_loss(torch.cat(
            #     [z_train_t, torch.cat(z_train_aug_t_level, dim=0), torch.cat(z_train_aug_t_length, dim=0)],
            #     dim=0), torch.cat(
            #     [z_valid_t, torch.cat(z_valid_aug_t_level, dim=0), torch.cat(z_valid_aug_t_length, dim=0)],
            #     dim=0)).item()
            # total_loss.append(loss)

            total_loss['level'] = dict()
            f1score['level'] = dict()
            for i, train_level in enumerate(train_levels):
                total_loss['level'][train_level] = dict()
                f1score['level'][train_level] = dict()
                classify_model = None
                for j, valid_level in enumerate(valid_levels):
                    loss = -W_loss(z_train_aug_t_level[i], z_valid_aug_t_level[j]).item()
                    total_loss['level'][train_level][valid_level] = loss
                    loss = -loss
                    if classify_model == None:
                        X = np.concatenate([z_train_t.numpy(), z_train_aug_t_level[i].numpy()], axis=0)
                        y = np.concatenate(
                            [np.zeros((len(train_normal_index), x_train_np.shape[1])), train_labels_level[i]], axis=0)
                        classify_model = train_classify_model(torch.tensor(X).float().to('cuda:0'),
                                                              torch.tensor(y).float().to('cuda:0'))
                    X_valid = np.concatenate([z_valid_t, z_valid_aug_t_level[j]], axis=0)
                    y_pred = classify(classify_model, torch.tensor(X_valid).float().to('cuda:0'))
                    y_valid = np.concatenate(
                        [np.zeros((len(valid_normal_index), x_valid_np.shape[1])), valid_labels_level[j]], axis=0)
                    f1 = f1_score(y_valid.reshape(-1), y_pred.reshape(-1))
                    f1score['level'][train_level][valid_level] = f1
                    print(f'train_level: {train_level}, valid_level: {valid_level}, wd: {loss}, f1score: {f1}')

            total_loss['length'] = dict()
            f1score['length'] = dict()
            for i, train_length in enumerate(train_lengths):
                total_loss['length'][train_length] = dict()
                f1score['length'][train_length] = dict()
                classify_model = None
                for j, valid_length in enumerate(valid_lengths):
                    loss = -W_loss(z_train_aug_t_length[i], z_valid_aug_t_length[j]).item()
                    total_loss['length'][train_length][valid_length] = loss
                    loss = -loss
                    if classify_model == None:
                        X = np.concatenate([z_train_t.numpy(), z_train_aug_t_length[i].numpy()], axis=0)
                        y = np.concatenate(
                            [np.zeros((len(train_normal_index), x_train_np.shape[1])), train_labels_length[i]], axis=0)
                        classify_model = train_classify_model(torch.tensor(X).float().to('cuda:0'),
                                                              torch.tensor(y).float().to('cuda:0'))
                    X_valid = np.concatenate([z_valid_t, z_valid_aug_t_length[j]], axis=0)
                    y_pred = classify(classify_model, torch.tensor(X_valid).float().to('cuda:0'))
                    y_valid = np.concatenate(
                        [np.zeros((len(valid_normal_index), x_valid_np.shape[1])), valid_labels_length[j]], axis=0)
                    f1 = f1_score(y_valid.reshape(-1), y_pred.reshape(-1))
                    f1score['length'][train_length][valid_length] = f1
                    print(f'train_length: {train_length}, valid_length: {valid_length}, wd: {loss}, f1score: {f1}')

            # z_test_t = emb.normalize(z_test)
            # X = np.concatenate([z_train_t.numpy(), z_aug_t.numpy()], axis=0)
            # y = np.concatenate([np.zeros((len(train_normal_index), x_train_np.shape[1])), labels], axis=0)
            # y_pred = classify(torch.tensor(X).float().to('cuda:0'), torch.tensor(y).float().to('cuda:0'),
            #                   z_test_t.to('cuda:0'))
            # f1score.append(f1_score(y_test.reshape(-1), y_pred.reshape(-1)))

            # X = np.concatenate([z_train_t.numpy(), torch.cat(z_train_aug_t_level, dim=0).numpy(),
            #                     torch.cat(z_train_aug_t_length, dim=0).numpy()], axis=0)
            # y = np.concatenate(
            #     [np.zeros((len(train_normal_index), x_train_np.shape[1])),
            #      np.concatenate(train_labels_level, axis=0),
            #      np.concatenate(train_labels_length, axis=0)], axis=0)
            # y_pred = classify(torch.tensor(X).float().to('cuda:0'), torch.tensor(y).float().to('cuda:0'),
            #                   torch.tensor(np.concatenate([z_valid_t, np.concatenate(z_valid_aug_t_level, axis=0),
            #                                                np.concatenate(z_valid_aug_t_length, axis=0)],
            #                                               axis=0)).float().to('cuda:0'))
            # y_valid = np.concatenate(
            #     [np.zeros((len(valid_normal_index), x_valid_np.shape[1])), np.concatenate(valid_labels_level, axis=0),
            #      np.concatenate(valid_labels_length, axis=0)], axis=0)
            # f1score.append(f1_score(y_valid.reshape(-1), y_pred.reshape(-1)))

            visualize(z_train, z_valid, z_train_aug_level, z_valid_aug_level, train_levels, valid_levels,
                      'level', f'length{fixed_length}', 1)
            visualize(z_train, z_valid, z_train_aug_length, z_valid_aug_length, train_lengths,
                      valid_lengths, 'length', f'level{fixed_level}', 1)
            visualize(z_train, z_valid, z_train_aug_level, z_valid_aug_level, train_levels, valid_levels,
                      'level', f'length{fixed_length}', -1)
            visualize(z_train, z_valid, z_train_aug_length, z_valid_aug_length, train_lengths,
                      valid_lengths, 'length', f'level{fixed_level}', -1)

            # visualize(z_train_aug_level, z_valid_aug_level, train_levels, valid_levels, 'level',
            #           f'length{fixed_length}', 1)
            # visualize(z_train_aug_length, z_valid_aug_length, train_lengths, valid_lengths, 'length',
            #           f'level{fixed_level}', 1)
            # visualize(z_train_aug_level, z_valid_aug_level, train_levels, valid_levels, 'level',
            #           f'length{fixed_length}', -1)
            # visualize(z_train_aug_length, z_valid_aug_length, train_lengths, valid_lengths, 'length',
            #           f'level{fixed_level}', -1)

    return total_loss, f1score


# def visualize(train_aug, valid_aug, train_values, test_values, para_name, fixed_value, converse=1):
#     all_data = torch.cat(valid_aug, dim=0).cpu().numpy()
#     xt = TSNE(n_components=2, random_state=42).fit_transform(all_data)
#     plt.figure(figsize=(12, 8))
#
#     # train anomalies
#     cmap = cm.get_cmap('Greys')
#     normalized_values = (train_values - np.min(train_values)) / (np.max(train_values) - np.min(train_values))
#     colors = [cmap(val) for val in normalized_values]
#     for i, value in enumerate(train_values):
#         start_idx = i * len(train_aug[i]) // len(train_values)
#         end_idx = start_idx + len(train_aug[i]) // len(train_values)
#         if converse == 1:
#             plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                         c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
#                         label=f'Anomalies (Platform) ({para_name}={value})')
#         else:
#             plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                         c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
#                         zorder=len(train_values) - i,
#                         label=f'Anomalies (Platform) ({para_name}={value})')
#     legend_elements = []
#     for i, value in enumerate(train_values):
#         legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Anomalies ({para_name}={value})',
#                                       markerfacecolor=colors[i], markersize=10))
#
#     # test anomalies
#     cmap = cm.get_cmap('Reds')
#     normalized_values = (test_values - np.min(test_values)) / (np.max(test_values) - np.min(test_values))
#     colors = [cmap(val) for val in normalized_values]
#     for i, value in enumerate(test_values):
#         start_idx = i * len(valid_aug[i]) // len(test_values)
#         end_idx = start_idx + len(valid_aug[i]) // len(test_values)
#         if converse == 1:
#             plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                         c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
#                         label=f'Anomalies (Platform) ({para_name}={value})')
#         else:
#             plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                         c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
#                         zorder=len(test_values) - i,
#                         label=f'Anomalies (Platform) ({para_name}={value})')
#     for i, value in enumerate(test_values):
#         legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Anomalies ({para_name}={value})',
#                                       markerfacecolor=colors[i], markersize=10))
#     plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
#     plt.xlabel('t-SNE 1')
#     plt.ylabel('t-SNE 2')
#     plt.title('t-SNE Visualization of Embeddings')
#     plt.tight_layout()
#     plt.savefig(f'logs/training/grid/{fixed_value}_{converse}_embedding_visualization_wo_normal.pdf',
#                 dpi=300,bbox_inches='tight')
#     plt.close()


def visualize(train_normal, test_normal, train_aug_list, aug_list,
              train_values, test_values, para_name, fixed_value, converse=1):
    aug = torch.cat(aug_list, dim=0)
    all_data = torch.cat([train_normal, test_normal, aug], dim=0).cpu().numpy()
    xt = TSNE(n_components=2, random_state=42).fit_transform(all_data)
    plt.figure(figsize=(12, 8))
    # train normal
    plt.scatter(xt[:len(train_normal), 0], xt[:len(train_normal), 1], c='b', alpha=0.5, label='Train Normal')
    # test normal
    plt.scatter(xt[len(train_normal):len(train_normal) + len(test_normal), 0],
                xt[len(train_normal):len(train_normal) + len(test_normal), 1],
                c='g', alpha=0.5, label='Test Normal')

    # train anomalies
    cmap = cm.get_cmap('Greys')
    normalized_values = (train_values - np.min(train_values)) / (np.max(train_values) - np.min(train_values))
    colors = [cmap(val) for val in normalized_values]
    for i, value in enumerate(train_values):
        start_idx = len(train_normal) + len(test_normal) + i * len(train_aug_list[i]) // len(train_values)
        end_idx = start_idx + len(train_aug_list[i]) // len(train_values)
        if converse == 1:
            plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
                        c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
                        label=f'Anomalies (Platform) ({para_name}={value})')
        else:
            plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
                        c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
                        zorder=len(train_values) - i,
                        label=f'Anomalies (Platform) ({para_name}={value})')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Train Normal', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Test Normal', markerfacecolor='g', markersize=10),
    ]
    for i, value in enumerate(train_values):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Anomalies ({para_name}={value})',
                                      markerfacecolor=colors[i], markersize=10))

    # test anomalies
    cmap = cm.get_cmap('Reds')
    normalized_values = (test_values - np.min(test_values)) / (np.max(test_values) - np.min(test_values))
    colors = [cmap(val) for val in normalized_values]
    for i, value in enumerate(test_values):
        start_idx = len(train_normal) + len(test_normal) + i * len(aug_list[i]) // len(test_values)
        end_idx = start_idx + len(aug_list[i]) // len(test_values)
        if converse == 1:
            plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
                        c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
                        label=f'Anomalies (Platform) ({para_name}={value})')
        else:
            plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
                        c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
                        zorder=len(test_values) - i,
                        label=f'Anomalies (Platform) ({para_name}={value})')
    for i, value in enumerate(test_values):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Anomalies ({para_name}={value})',
                                      markerfacecolor=colors[i], markersize=10))
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization of Embeddings')
    plt.tight_layout()
    plt.savefig(f'logs/training/grid/{fixed_value}_{converse}_embedding_visualization.pdf', dpi=300,
                bbox_inches='tight')
    plt.close()


# def visualize_fixed_grid(train_normal, test_normal, train_aug, aug_list,
#                          train_value, test_values, para_name, fixed_value, converse=1):
#     aug = torch.cat(aug_list, dim=0)
#     all_data = torch.cat([train_normal, test_normal, aug], dim=0).cpu().numpy()
#     xt = TSNE(n_components=2, random_state=42).fit_transform(all_data)
#     plt.figure(figsize=(12, 8))
#
#     # train normal data
#     plt.scatter(xt[:len(train_normal), 0], xt[:len(train_normal), 1], c='b', alpha=0.5, label='Train Normal')
#
#     # test normal data
#     plt.scatter(xt[len(train_normal):len(train_normal) + len(test_normal), 0],
#                 xt[len(train_normal):len(train_normal) + len(test_normal), 1],
#                 c='g', alpha=0.5, label='Test Normal')
#
#     # train anomalies
#     cmap_train = cm.get_cmap('Greys')
#     normalized_train_value = (train_value - np.min(train_value)) / (np.max(train_value) - np.min(train_value))
#     train_color = cmap_train(normalized_train_value)
#     start_idx = len(train_normal) + len(test_normal)
#     end_idx = start_idx + len(train_aug)
#     plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                 c=[train_color] * len(train_aug), alpha=0.8,
#                 label=f'Train Anomaly ({para_name}={train_value})')
#
#     # test anomalies
#     cmap_test = cm.get_cmap('Reds')
#     normalized_test_values = (test_values - np.min(test_values)) / (np.max(test_values) - np.min(test_values))
#     test_colors = [cmap_test(val) for val in normalized_test_values]
#     for i, value in enumerate(test_values):
#         start_idx = len(train_normal) + len(test_normal) + i * len(aug_list[i]) // len(test_values)
#         end_idx = start_idx + len(aug_list[i]) // len(test_values)
#         plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                     c=[test_colors[i]] * (end_idx - start_idx), alpha=0.5,
#                     zorder=len(test_values) - i if converse != 1 else None,
#                     label=f'Test Anomalies ({para_name}={value})')
#
#     # legend
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='Train Normal', markerfacecolor='b', markersize=10),
#         Line2D([0], [0], marker='o', color='w', label='Test Normal', markerfacecolor='g', markersize=10)
#     ]
#     for i, value in enumerate(train_values):
#         legend_elements.append(
#             Line2D([0], [0], marker='o', color='w', label=f'Train Anomalies ({para_name}={value})',
#                    markerfacecolor=train_colors[i], markersize=10)
#         )
#     for i, value in enumerate(test_values):
#         legend_elements.append(
#             Line2D([0], [0], marker='o', color='w', label=f'Test Anomalies ({para_name}={value})',
#                    markerfacecolor=test_colors[i], markersize=10)
#         )
#
#     plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
#     plt.xlabel('t-SNE 1')
#     plt.ylabel('t-SNE 2')
#     plt.title(f't-SNE Visualization (Fixed {para_name}={fixed_value})')
#     plt.tight_layout()
#
#     plt.savefig(f'logs/training/fixed_grid/{fixed_value}_{converse}_embedding_visualization.pdf', dpi=300,
#                 bbox_inches='tight')
#     plt.close()

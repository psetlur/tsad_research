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
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import random
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



def hist_sample(cdf, bins):
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    val = np.random.uniform(bins[bin_idx - 1], bins[bin_idx])
    return val


def black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, a_config):
    n_trials = 1
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

        emb = EmbNormalizer()
        total_loss = dict()
        f1score = dict()
        for seed in range(n_trials):
            train_inlier_index, train_outlier_index = train_test_split(range(len(x_train_np)),
                                                                       train_size=1 - ratio_anomaly, random_state=seed)

            valid_inlier_index, valid_outlier_index = train_test_split(range(len(x_valid_np)),
                                                                       train_size=1 - ratio_anomaly, random_state=seed)


            def inject(x_np, configs, outlier_index, config_name):
                x_configs_augs, configs_labels = list(), list()
                for config in configs:
                    x_augs, labels = list(), list()
                    for i in outlier_index:
                        if config_name == 'level':
                            x_aug, label = inject_platform(x_np[i], config, np.random.uniform(0, 0.5), fixed_length)
                        elif config_name == 'length':
                            x_aug, label = inject_platform(x_np[i], fixed_level, np.random.uniform(0, 0.5), config)
                        else:
                            raise Exception('Unsupported config')
                        x_augs.append(x_aug)
                        labels.append(label)
                    x_configs_augs.append(x_augs)
                    configs_labels.append(labels)
                return x_configs_augs, configs_labels

            # train level aug
            x_train_level_aug, train_level_labels = inject(x_np=x_train_np, configs=train_levels,
                                                           outlier_index=train_outlier_index, config_name='level')
            # train length aug
            x_train_length_aug, train_length_labels = inject(x_np=x_train_np, configs=train_lengths,
                                                             outlier_index=train_outlier_index, config_name='length')
            # valid level aug
            x_valid_level_aug, valid_level_labels = inject(x_np=x_valid_np, configs=valid_levels,
                                                           outlier_index=valid_outlier_index, config_name='level')
            # valid length aug
            x_valid_length_aug, valid_length_labels = inject(x_np=x_valid_np, configs=valid_lengths,
                                                             outlier_index=valid_outlier_index, config_name='length')

            # x_aug, labels = [], []
            # for i in train_outlier_index:
            #     x = x_train_np[i]
            #     xa, l = inject_platform(x, fixed_level, np.random.uniform(0, 0.5), fixed_length)
            #     x_aug.append(xa)
            #     labels.append(l)

            z_train_level_aug = [
                model(torch.tensor(np.array(level_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
                level_x_aug in x_train_level_aug]
            z_train_length_aug = [
                model(torch.tensor(np.array(length_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
                length_x_aug in x_train_length_aug]
            z_valid_level_aug = [
                model(torch.tensor(np.array(level_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
                level_x_aug in x_valid_level_aug]
            z_valid_length_aug = [
                model(torch.tensor(np.array(length_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
                length_x_aug in x_valid_length_aug]
            
            train_augs = {"level": z_train_level_aug, "length": z_train_length_aug}
            valid_augs = {"level": z_valid_level_aug, "length": z_valid_length_aug}
            train_configs = {"level": train_levels, "length": train_lengths}
            valid_configs = {"level": valid_levels, "length": valid_lengths}

            # z_aug = model(torch.tensor(np.array(x_aug)).float().unsqueeze(1).to(args.device)).detach()
            
            # visualize(train_inlier=z_train, valid_inlier=z_valid, train_augs=z_train_level_aug,
            #           valid_augs=z_valid_level_aug, train_configs=train_levels, valid_configs=valid_levels,
            #           config_name='level', fixed_config=f'fixed_length{fixed_length}', trail=args.trail, test=True)

            visualize_multiple(train_inlier = z_train, valid_inlier = z_valid, train_augs = train_augs, valid_augs = valid_augs,
                               train_configs = train_configs, valid_configs = valid_configs, config_name = 'level',
                               fixed_config = f'fixed_length{fixed_length}', trail = args.trail, inlier = True, test = True)

            # # with test
            # visualize(train_inlier=z_train, valid_inlier=z_valid, train_augs=z_train_level_aug,
            #           valid_augs=z_valid_level_aug, train_configs=train_levels, valid_configs=valid_levels,
            #           config_name='level', fixed_config=f'fixed_length{fixed_length}', trail=args.trail,test=True)
            # visualize(train_inlier=z_train, valid_inlier=z_valid, train_augs=z_train_length_aug,
            #           valid_augs=z_valid_length_aug, train_configs=train_lengths, valid_configs=valid_lengths,
            #           config_name='length', fixed_config=f'fixed_level{fixed_level}', trail=args.trail,test=True)
            #
            # # without test
            # visualize(train_inlier=z_train, valid_inlier=z_valid, train_augs=z_train_level_aug,
            #           valid_augs=None, train_configs=train_levels, valid_configs=None,
            #           config_name='level', fixed_config=f'fixed_length{fixed_length}', trail=args.trail,test=False)
            # visualize(train_inlier=z_train, valid_inlier=z_valid, train_augs=z_train_length_aug,
            #           valid_augs=None, train_configs=train_lengths, valid_configs=None,
            #           config_name='length', fixed_config=f'fixed_level{fixed_level}', trail=args.trail,test=False)

    return total_loss, f1score


def visualize(train_inlier, valid_inlier, train_augs, valid_augs, train_configs, valid_configs, config_name,
              fixed_config, trail, inlier=True, test=True):
    train_aug = torch.cat(train_augs, dim=0)
    if test is True:
        valid_aug = torch.cat(valid_augs, dim=0)
        aug = torch.cat([train_aug, valid_aug], dim=0)
    else:
        aug = torch.cat(train_augs, dim=0)

    all_data = torch.cat([train_inlier, valid_inlier, aug], dim=0).to('cpu').numpy()
    inlier_size = len(train_inlier) + len(valid_inlier)

    xt = TSNE(n_components=2, random_state=42).fit_transform(all_data)

    if config_name == 'level':
        plt.figure(figsize=(12, 10))
    elif config_name == 'length':
        plt.figure(figsize=(12, 8))
    else:
        raise Exception('Unsupported config_name.')
    # if inlier is True:
    # train inliers
    plt.scatter(xt[:len(train_inlier), 0], xt[:len(train_inlier), 1], c='b', alpha=0.5)
    # test inliers
    plt.scatter(xt[len(train_inlier):len(train_inlier) + len(valid_inlier), 0],
                xt[len(train_inlier):len(train_inlier) + len(valid_inlier), 1], c='g', alpha=0.5)

    # train outliers
    if test is True:
        cmap = cm.get_cmap('Greys')
    else:
        cmap = cm.get_cmap('Reds')

    #     cmap = cm.get_cmap('tab20')
    if len(train_configs) == 1:
        normalized_values = [0.5]
    else:
        normalized_values = (train_configs - np.min(train_configs)) / (np.max(train_configs) - np.min(train_configs))
        normalized_values = normalized_values * (1 - 0.1) + 0.1
    colors = [cmap(val) for val in normalized_values]
    for i, value in enumerate(train_configs):
        start_idx = inlier_size + i * len(train_augs[i])
        end_idx = start_idx + len(train_augs[i])
        plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1], c=[colors[i]] * (end_idx - start_idx),
                    alpha=0.5)
    
    legend_elements = list()
    # if inlier is True:
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label='Train inliers', markerfacecolor='b', markersize=10))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label='Test inliers', markerfacecolor='g', markersize=10))
    for i, value in enumerate(train_configs):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Train outliers ({config_name}={value})',
                                      markerfacecolor=colors[i], markersize=10))

    # valid outliers
    if test is True:
        cmap = cm.get_cmap('Reds')
        if len(valid_configs) == 1:
            normalized_values = [0.5]
        else:
            normalized_values = (valid_configs - np.min(valid_configs)) / (
                    np.max(valid_configs) - np.min(valid_configs))
            normalized_values = normalized_values * (1 - 0.1) + 0.1
        colors = [cmap(val) for val in normalized_values]
        for i, value in enumerate(valid_configs):
            start_idx = inlier_size + len(train_aug) + i * len(valid_augs[i])
            end_idx = start_idx + len(valid_augs[i])
            plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1], c=[colors[i]] * (end_idx - start_idx),
                        alpha=0.5)
            
        for i, value in enumerate(valid_configs):
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', label=f'Test outliers ({config_name}={value})',
                       markerfacecolor=colors[i], markersize=10))

    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization of Embeddings')

    plt.tight_layout()
    # plt.savefig(f'logs/training/{trail}/{fixed_config}_inlier{inlier}_test{test}.pdf', bbox_inches='tight')
    plt.savefig(f'logs/training/{trail}/{fixed_config}_test{test}.pdf', bbox_inches='tight')
    plt.close()


def visualize_multiple(train_inlier, valid_inlier, train_augs, valid_augs, train_configs, valid_configs, config_name,
                       fixed_config, trail, inlier = True, test = True):
    train_aug = torch.cat([torch.cat(v, dim = 0) for v in train_augs.values()], dim = 0)
    if test is True:
        valid_aug = torch.cat([torch.cat(v, dim = 0) for v in valid_augs.values()], dim = 0)
        aug = torch.cat([train_aug, valid_aug], dim = 0)
    else:
        aug = torch.cat(train_augs, dim = 0)

    all_data = torch.cat([train_inlier, valid_inlier, aug], dim = 0).to('cpu').numpy()
    inlier_size = len(train_inlier) + len(valid_inlier)

    xt = TSNE(n_components = 2, random_state = 42).fit_transform(all_data)
    plt.figure(figsize = (12, 10))

    # train and test inliers
    plt.scatter(xt[:len(train_inlier), 0], xt[:len(train_inlier), 1], c = 'b', alpha = 0.5, label = 'Train inliers')
    plt.scatter(xt[len(train_inlier) : len(train_inlier) + len(valid_inlier), 0],
                xt[len(train_inlier) : len(train_inlier) + len(valid_inlier), 1], c = 'g', alpha = 0.5, label = 'Test inliers')
    
    anomaly_types = list(set(train_configs.keys())) # getting anomaly types
    colormap = cm.get_cmap('tab10', len(anomaly_types)) # different colors for each anomaly

    legend_elements = [
        Line2D([0], [0], marker = 'o', color = 'w', label = 'Train inliers', markerfacecolor = 'b', markersize = 10),
        Line2D([0], [0], marker = 'o', color = 'w', label = 'Test inliers', markerfacecolor = 'g', markersize = 10)
    ]

    start_idx = inlier_size
    for idx, anomaly_type in enumerate(anomaly_types):
        cmap = cm.get_cmap('coolwarm') # different colors for intra-anomaly variation
        config_values = np.array(train_configs[anomaly_type])
        normalized_values = (config_values - np.min(config_values)) / (np.max(config_values) - np.min(config_values))
        normalized_values = normalized_values * (1 - 0.1) + 0.1
        colors = [cmap(val) for val in normalized_values]

        for i, value in enumerate(config_values):
            end_idx = start_idx + len(train_augs[anomaly_type][i])
            plt.scatter(xt[start_idx : end_idx, 0], xt[start_idx : end_idx, 1], c = [colors[i]] * (end_idx - start_idx), alpha = 0.5)
            start_idx = end_idx
        
        legend_elements.append(Line2D([0], [0], marker = 'o', color = 'w',
                                      label = f'Train outliers ({anomaly_type})', markerfacecolor = colors[0], markersize = 10))
        
    if test is True:
        for idx, anomaly_type in enumerate(anomaly_types):
            cmap = cm.get_cmap('Reds')
            config_values = np.array(valid_configs[anomaly_type])
            normalized_values = (config_values - np.min(config_values)) / (np.max(config_values) - np.min(config_values))
            normalized_values = normalized_values * (1 - 0.1) + 0.1
            colors = [cmap(val) for val in normalized_values]

            for i, value in enumerate(config_values):
                end_idx = start_idx + len(valid_augs[anomaly_type][i])
                plt.scatter(xt[start_idx : end_idx, 0], xt[start_idx : end_idx, 1], c = [colors[i]] * (end_idx - start_idx), alpha = 0.5)
                start_idx = end_idx
            
            legend_elements.append(Line2D([0], [0], marker = 'o', color = 'w',
                                          label = f'Test outliers ({anomaly_type})', markerfacecolor = colors[0], markersize = 10))
    
    plt.legend(handles = legend_elements, loc = "upper left", bbox_to_anchor = (1, 1))
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization of Embeddings')
    plt.tight_layout()
    plt.savefig(f'logs/training/{trail}/{fixed_config}_test{test}.pdf', bbox_inches = 'tight')
    plt.close()

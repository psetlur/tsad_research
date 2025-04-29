import logging
import math
import os
from geomloss import SamplesLoss
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from utils.utils import EarlyStopping

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)


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
    end_a = min(start_a + length_a, len(ts_row))  # Ensure end index is valid
    ts_row[start_a: end_a] = float(level)
    label[start_a: end_a] = 1
    return ts_row, label


def inject_mean(ts_row, level, start, length):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    start_a = int(len(ts_row) * start)
    length_a = int(len(ts_row) * length)
    end_a = min(start_a + length_a, len(ts_row))
    ts_row[start_a: end_a] += float(level)
    label[start_a: end_a] = 1
    return ts_row, label


def inject_spike(ts_row, level, start):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    start_a = int(len(ts_row) * start)
    ts_row[start_a] = ts_row[start_a] + level if np.random.rand() < 0.5 else ts_row[start_a] - level
    label[start_a] = 1
    return ts_row, label


def inject_spike_train(ts_row, level):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    s = set()
    for _ in range(len(ts_row) // len(ts_row)):
        start_a = int(len(ts_row) * np.random.uniform(0, 1))
        while start_a in s:
            start_a = int(len(ts_row) * np.random.uniform(0, 1))
        s.add(start_a)
        ts_row[start_a] = ts_row[start_a] + level if np.random.rand() < 0.5 else ts_row[start_a] - level
        label[start_a] = 1
    return ts_row, label


def inject_amplitude(ts_row, level, start, length):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    start_a = int(len(ts_row) * start)
    length_a = int(len(ts_row) * length)
    end_a = min(start_a + length_a, len(ts_row))
    actual_length = end_a - start_a
    if actual_length > 0:
        amplitude_bell = np.ones(actual_length) * level
        ts_row[start_a: end_a] *= amplitude_bell
        label[start_a: end_a] = 1
    return ts_row, label


def inject_trend(ts_row, slope, start, length):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    ts_len = len(ts_row)
    start = max(0.0, min(start, 1.0))
    length = max(0.0, min(length, 1.0))
    start_a = int(ts_len * start)
    length_a = int(ts_len * length)
    end_a = min(start_a + length_a, ts_len)
    actual_length = end_a - start_a
    if actual_length <= 0:
        return ts_row, label
    slope_a = np.arange(0, actual_length) * slope
    ts_row[start_a: end_a] += slope_a
    if slope_a.size > 0 and end_a < ts_len:
        ts_row[end_a:] += np.full(ts_len - end_a, slope_a[-1])
    label[start_a: end_a] = 1
    return ts_row, label


def inject_variance(ts_row, level, start, length):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    start_a = int(len(ts_row) * start)
    length_a = int(len(ts_row) * length)
    end_a = min(start_a + length_a, len(ts_row))
    actual_length = end_a - start_a
    if actual_length > 0:
        var = np.random.normal(0, level, actual_length)
        ts_row[start_a: end_a] += var
        label[start_a: end_a] = 1
    return ts_row, label


def train_classify_model(args, X_train, y_train, sequence_length=512):
    embed_dim = X_train.shape[1]
    epoch = 100000
    model = nn.Sequential(
        nn.Linear(embed_dim, 128),
        nn.ReLU(),
        nn.Linear(128, sequence_length)
    ).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    early_stopping = EarlyStopping(1000)

    train_index, test_index = train_test_split(range(len(X_train)), train_size=1 - 0.1, random_state=0)
    for i in range(epoch):
        out = model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(X_train[test_index])
            loss = criterion(out, y_train[test_index]).item()
            if early_stopping(loss):
                print(f'Early Stopping at epoch: {i}')
                break
    return model


def classify(model, X_valid):
    model.eval()
    with torch.no_grad():
        logits = model(X_valid)
        probs = torch.sigmoid(logits)
        y_pred = torch.where(probs > 0.5, 1, 0).cpu().numpy()
    return y_pred


def black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point=None,
                       valid_anomaly_types=None, train_point=None, best=False, calculate_f1=True):
    ratio_anomaly = 0.1
    min_platform_level = -1.0
    min_platform_length = 0.2
    max_platform_level = 1.1
    max_platform_length = 0.52
    fixed_platform_level = 0.5
    fixed_platform_length = 0.3
    platform_level_step = 0.1
    platform_length_step = 0.02

    min_mean_level = -1.0
    min_mean_length = 0.2
    max_mean_level = 1.1
    max_mean_length = 0.52
    fixed_mean_level = 0.5
    fixed_mean_length = 0.3
    mean_level_step = 0.1
    mean_length_step = 0.02

    min_spike_level = 2
    max_spike_level = 22
    fixed_spike_level = 2
    spike_level_step = 2

    min_amplitude_level = [0.1, 2]
    min_amplitude_length = 0.2
    max_amplitude_level = [1, 11]
    max_amplitude_length = 0.52
    fixed_amplitude_level = [0.1, 10]
    fixed_amplitude_length = 0.3
    amplitude_level_step = [0.1, 1]
    amplitude_length_step = 0.02

    min_trend_slope = [-0.01, 0.001, 5]
    min_trend_length = 0.2
    max_trend_slope = [0, 0.011, 25]
    max_trend_length = 0.52
    fixed_trend_slope = 0.01
    fixed_trend_length = 0.3
    trend_slope_step = [0.001, 5]
    trend_length_step = 0.02

    min_variance_level = 0.1
    min_variance_length = 0.2
    max_variance_level = 0.51
    max_variance_length = 0.52
    fixed_variance_level = 0.1
    fixed_variance_length = 0.3
    variance_level_step = 0.05
    variance_length_step = 0.02

    anomaly_types = ['platform', 'mean', 'spike', 'amplitude', 'trend', 'variance']
    # anomaly_types = ['platform']
    # anomaly_types = ['mean']
    # anomaly_types = ['spike']
    # anomaly_types = ['amplitude']
    # anomaly_types = ['trend']
    # anomaly_types = ['variance']
    model.eval()

    with torch.no_grad():
        z_train_list, x_train_list = [], []
        for x_batch in train_dataloader:
            c_x = model(x_batch.to(args.device)).detach()
            z_train_list.append(c_x)
            x_train_list.append(x_batch.cpu().numpy())  # Move to CPU before numpy
        z_train = torch.cat(z_train_list, dim=0)
        x_train_np = np.concatenate(x_train_list, axis=0)
        if x_train_np.ndim == 3 and x_train_np.shape[1] == 1:
            x_train_np = x_train_np.squeeze(1)
        elif x_train_np.ndim != 2:
            raise ValueError(f"Unexpected shape for x_train_np: {x_train_np.shape}")
        sequence_length = x_train_np.shape[1]  # Get sequence length

        z_valid_list, x_valid_list = [], []
        for x_batch in val_dataloader:
            c_x = model(x_batch.to(args.device)).detach()
            z_valid_list.append(c_x)
            x_valid_list.append(x_batch.cpu().numpy())
        z_valid = torch.cat(z_valid_list, dim=0)
        x_valid_np = np.concatenate(x_valid_list, axis=0)
        if x_valid_np.ndim == 3 and x_valid_np.shape[1] == 1:
            x_valid_np = x_valid_np.squeeze(1)  # Reshape to [N, SeqLen]
        elif x_valid_np.ndim != 2:
            raise ValueError(f"Unexpected shape for x_valid_np: {x_valid_np.shape}")

    train_inlier_index, train_outlier_index = train_test_split(range(len(x_train_np)),
                                                               train_size=1 - ratio_anomaly, random_state=0)
    valid_inlier_index, valid_outlier_index = train_test_split(range(len(x_valid_np)),
                                                               train_size=1 - ratio_anomaly, random_state=0)
    emb = EmbNormalizer()
    if valid_point == None:
        x_augs = list()
        for index, anomaly_type in enumerate(anomaly_types):
            for i in train_outlier_index[index * len(train_outlier_index) // len(anomaly_types):(index + 1) * len(
                    train_outlier_index) // len(anomaly_types)]:
                if anomaly_type == 'platform':
                    x_aug, _ = inject_platform(x_train_np[i], fixed_platform_level, np.random.uniform(0, 0.5),
                                               fixed_platform_length)
                elif anomaly_type == 'mean':
                    x_aug, _ = inject_mean(x_train_np[i], fixed_mean_level, np.random.uniform(0, 0.5),
                                           fixed_mean_length)
                elif anomaly_type == 'spike':
                    # x_aug, _ = inject_spike(x_train_np[i], fixed_spike_level, np.random.uniform(0, 1))
                    x_aug, _ = inject_spike_train(x_train_np[i], fixed_spike_level)
                    # visualize_time_series(x_aug)
                    # print()
                elif anomaly_type == 'amplitude':
                    x_aug, _ = inject_amplitude(x_train_np[i], fixed_amplitude_level[1], np.random.uniform(0, 0.5),
                                                fixed_amplitude_length)
                elif anomaly_type == 'trend':
                    x_aug, _ = inject_trend(x_train_np[i], fixed_trend_slope, np.random.uniform(0, 0.5),
                                            fixed_trend_length)
                elif anomaly_type == 'variance':
                    x_aug, _ = inject_variance(x_train_np[i], fixed_variance_level, np.random.uniform(0, 0.5),
                                               fixed_variance_length)
                else:
                    raise Exception('Unsupported anomaly_type.')
                x_augs.append(x_aug)

        emb = EmbNormalizer()
        z_aug = model(torch.tensor(np.array(x_augs)).float().unsqueeze(1).to(0)).detach()
        z_train_t, z_valid_t, _ = emb(z_train[train_inlier_index].clone().squeeze(),
                                      z_valid[valid_inlier_index].clone().squeeze(), z_aug)

        def argument(x_np, outlier_index, is_train=False):
            x_augs_dict, labels_dict = dict(), dict()
            for index, anomaly_type in enumerate(anomaly_types):
                x_augs_dict[anomaly_type] = dict()
                labels_dict[anomaly_type] = dict()
                for i in outlier_index[index * (len(outlier_index) // len(anomaly_types)):(index + 1) * (len(
                        outlier_index) // len(anomaly_types))]:
                    x = x_np[i]
                    if anomaly_type == 'platform':
                        if x_augs_dict[anomaly_type].get('level') is None:
                            x_augs_dict[anomaly_type]['level'] = dict()
                            labels_dict[anomaly_type]['level'] = dict()
                            x_augs_dict[anomaly_type]['length'] = dict()
                            labels_dict[anomaly_type]['length'] = dict()
                        for level in np.round(np.arange(min_platform_level, max_platform_level, platform_level_step),
                                              1):
                            if x_augs_dict[anomaly_type]['level'].get(level) is None:
                                x_augs_dict[anomaly_type]['level'][level] = list()
                                labels_dict[anomaly_type]['level'][level] = list()
                            x_aug, label = inject_platform(x, level, np.random.uniform(0, 0.5), fixed_platform_length)
                            x_augs_dict[anomaly_type]['level'][level].append(x_aug)
                            labels_dict[anomaly_type]['level'][level].append(label)
                        for length in np.round(
                                np.arange(min_platform_length, max_platform_length, platform_length_step), 2):
                            if x_augs_dict[anomaly_type]['length'].get(length) is None:
                                x_augs_dict[anomaly_type]['length'][length] = list()
                                labels_dict[anomaly_type]['length'][length] = list()
                            x_aug, label = inject_mean(x, fixed_mean_level, np.random.uniform(0, 0.5), length)
                            x_augs_dict[anomaly_type]['length'][length].append(x_aug)
                            labels_dict[anomaly_type]['length'][length].append(label)
                    elif anomaly_type == 'mean':
                        if x_augs_dict[anomaly_type].get('level') is None:
                            x_augs_dict[anomaly_type]['level'] = dict()
                            labels_dict[anomaly_type]['level'] = dict()
                            x_augs_dict[anomaly_type]['length'] = dict()
                            labels_dict[anomaly_type]['length'] = dict()
                        for level in np.round(np.arange(min_mean_level, max_mean_level, mean_level_step), 1):
                            if level == 0:
                                continue
                            if x_augs_dict[anomaly_type]['level'].get(level) is None:
                                x_augs_dict[anomaly_type]['level'][level] = list()
                                labels_dict[anomaly_type]['level'][level] = list()
                            x_aug, label = inject_mean(x, level, np.random.uniform(0, 0.5), fixed_mean_length)
                            x_augs_dict[anomaly_type]['level'][level].append(x_aug)
                            labels_dict[anomaly_type]['level'][level].append(label)
                        for length in np.round(np.arange(min_mean_length, max_mean_length, mean_length_step), 2):
                            if x_augs_dict[anomaly_type]['length'].get(length) is None:
                                x_augs_dict[anomaly_type]['length'][length] = list()
                                labels_dict[anomaly_type]['length'][length] = list()
                            x_aug, label = inject_mean(x, fixed_mean_level, np.random.uniform(0, 0.5), length)
                            x_augs_dict[anomaly_type]['length'][length].append(x_aug)
                            labels_dict[anomaly_type]['length'][length].append(label)
                    elif anomaly_type == 'spike':
                        if x_augs_dict[anomaly_type].get('level') is None:
                            x_augs_dict[anomaly_type]['level'] = dict()
                            labels_dict[anomaly_type]['level'] = dict()
                        for level in np.round(np.arange(min_spike_level, max_spike_level, spike_level_step), 1):
                            if x_augs_dict[anomaly_type]['level'].get(level) is None:
                                x_augs_dict[anomaly_type]['level'][level] = list()
                                labels_dict[anomaly_type]['level'][level] = list()
                            if is_train:
                                x_aug, label = inject_spike_train(x, level)
                                # x_aug, label = inject_spike(x, level, np.random.uniform(0, 1))
                            else:
                                x_aug, label = inject_spike(x, level, np.random.uniform(0, 1))
                            x_augs_dict[anomaly_type]['level'][level].append(x_aug)
                            labels_dict[anomaly_type]['level'][level].append(label)
                    elif anomaly_type == 'amplitude':
                        if x_augs_dict[anomaly_type].get('level') is None:
                            x_augs_dict[anomaly_type]['level'] = dict()
                            labels_dict[anomaly_type]['level'] = dict()
                            x_augs_dict[anomaly_type]['length'] = dict()
                            labels_dict[anomaly_type]['length'] = dict()
                        for level in np.round(
                                np.arange(min_amplitude_level[0], max_amplitude_level[0], amplitude_level_step[0]), 1):
                            if x_augs_dict[anomaly_type]['level'].get(level) is None:
                                x_augs_dict[anomaly_type]['level'][level] = list()
                                labels_dict[anomaly_type]['level'][level] = list()
                            x_aug, label = inject_amplitude(x, level, np.random.uniform(0, 0.5), fixed_amplitude_length)
                            x_augs_dict[anomaly_type]['level'][level].append(x_aug)
                            labels_dict[anomaly_type]['level'][level].append(label)
                        for level in np.arange(min_amplitude_level[1], max_amplitude_level[1], amplitude_level_step[1]):
                            if x_augs_dict[anomaly_type]['level'].get(level) is None:
                                x_augs_dict[anomaly_type]['level'][level] = list()
                                labels_dict[anomaly_type]['level'][level] = list()
                            x_aug, label = inject_amplitude(x, level, np.random.uniform(0, 0.5), fixed_amplitude_length)
                            x_augs_dict[anomaly_type]['level'][level].append(x_aug)
                            labels_dict[anomaly_type]['level'][level].append(label)
                        for length in np.round(
                                np.arange(min_amplitude_length, max_amplitude_length, amplitude_length_step), 2):
                            if x_augs_dict[anomaly_type]['length'].get(length) is None:
                                x_augs_dict[anomaly_type]['length'][length] = list()
                                labels_dict[anomaly_type]['length'][length] = list()
                            x_aug, label = inject_amplitude(x, fixed_amplitude_level[1], np.random.uniform(0, 0.5),
                                                            length)
                            x_augs_dict[anomaly_type]['length'][length].append(x_aug)
                            labels_dict[anomaly_type]['length'][length].append(label)
                    elif anomaly_type == 'trend':
                        if x_augs_dict[anomaly_type].get('slope') is None:
                            x_augs_dict[anomaly_type]['slope'] = dict()
                            labels_dict[anomaly_type]['slope'] = dict()
                            x_augs_dict[anomaly_type]['length'] = dict()
                            labels_dict[anomaly_type]['length'] = dict()
                        for slope in np.round(np.arange(min_trend_slope[0], max_trend_slope[0], trend_slope_step[0]),
                                              3):
                            if x_augs_dict[anomaly_type]['slope'].get(slope) is None:
                                x_augs_dict[anomaly_type]['slope'][slope] = list()
                                labels_dict[anomaly_type]['slope'][slope] = list()
                            x_aug, label = inject_trend(x, slope, np.random.uniform(0, 0.5), fixed_trend_length)
                            x_augs_dict[anomaly_type]['slope'][slope].append(x_aug)
                            labels_dict[anomaly_type]['slope'][slope].append(label)
                        for slope in np.round(np.arange(min_trend_slope[1], max_trend_slope[1], trend_slope_step[0]),
                                              3):
                            if x_augs_dict[anomaly_type]['slope'].get(slope) is None:
                                x_augs_dict[anomaly_type]['slope'][slope] = list()
                                labels_dict[anomaly_type]['slope'][slope] = list()
                            x_aug, label = inject_trend(x, slope, np.random.uniform(0, 0.5), fixed_trend_length)
                            x_augs_dict[anomaly_type]['slope'][slope].append(x_aug)
                            labels_dict[anomaly_type]['slope'][slope].append(label)
                        for slope in np.round(np.arange(min_trend_slope[2], max_trend_slope[2], trend_slope_step[1]),
                                              3):
                            if x_augs_dict[anomaly_type]['slope'].get(slope) is None:
                                x_augs_dict[anomaly_type]['slope'][slope] = list()
                                labels_dict[anomaly_type]['slope'][slope] = list()
                            x_aug, label = inject_trend(x, slope, np.random.uniform(0, 0.5), fixed_trend_length)
                            x_augs_dict[anomaly_type]['slope'][slope].append(x_aug)
                            labels_dict[anomaly_type]['slope'][slope].append(label)
                        for length in np.round(np.arange(min_trend_length, max_trend_length, trend_length_step), 2):
                            if x_augs_dict[anomaly_type]['length'].get(length) is None:
                                x_augs_dict[anomaly_type]['length'][length] = list()
                                labels_dict[anomaly_type]['length'][length] = list()
                            x_aug, label = inject_trend(x, fixed_trend_slope, np.random.uniform(0, 0.5), length)
                            x_augs_dict[anomaly_type]['length'][length].append(x_aug)
                            labels_dict[anomaly_type]['length'][length].append(label)
                    elif anomaly_type == 'variance':
                        if x_augs_dict[anomaly_type].get('level') is None:
                            x_augs_dict[anomaly_type]['level'] = dict()
                            labels_dict[anomaly_type]['level'] = dict()
                            x_augs_dict[anomaly_type]['length'] = dict()
                            labels_dict[anomaly_type]['length'] = dict()
                        for level in np.round(np.arange(min_variance_level, max_variance_level, variance_level_step),
                                              2):
                            if x_augs_dict[anomaly_type]['level'].get(level) is None:
                                x_augs_dict[anomaly_type]['level'][level] = list()
                                labels_dict[anomaly_type]['level'][level] = list()
                            x_aug, label = inject_variance(x, level, np.random.uniform(0, 0.5), fixed_variance_length)
                            x_augs_dict[anomaly_type]['level'][level].append(x_aug)
                            labels_dict[anomaly_type]['level'][level].append(label)
                        for length in np.round(
                                np.arange(min_variance_length, max_variance_length, variance_length_step), 2):
                            if x_augs_dict[anomaly_type]['length'].get(length) is None:
                                x_augs_dict[anomaly_type]['length'][length] = list()
                                labels_dict[anomaly_type]['length'][length] = list()
                            x_aug, label = inject_variance(x, fixed_variance_level, np.random.uniform(0, 0.5), length)
                            x_augs_dict[anomaly_type]['length'][length].append(x_aug)
                            labels_dict[anomaly_type]['length'][length].append(label)
                    else:
                        raise Exception(f'Unsupported anomaly_type {anomaly_type}')
            return x_augs_dict, labels_dict

        x_train_aug, train_labels = argument(x_np=x_train_np, outlier_index=train_outlier_index, is_train=True)
        x_valid_aug, valid_labels = argument(x_np=x_valid_np, outlier_index=valid_outlier_index)

        z_train_aug = {
            anomaly_type: {
                config: {c: model(torch.tensor(np.array(x_aug)).float().unsqueeze(1).to(args.device)).detach()
                         for c, x_aug in x_augs.items()}
                for config, x_augs in x_train_aug[anomaly_type].items()}
            for anomaly_type in anomaly_types}
        z_valid_aug = {
            anomaly_type: {
                config: {c: model(torch.tensor(np.array(x_aug)).float().unsqueeze(1).to(args.device)).detach()
                         for c, x_aug in x_augs.items()}
                for config, x_augs in x_valid_aug[anomaly_type].items()}
            for anomaly_type in anomaly_types}

        z_train_aug_t = {anomaly_type: {config: {c: emb.normalize(z_aug) for c, z_aug in z_augs.items()}
                                        for config, z_augs in z_train_aug[anomaly_type].items()}
                         for anomaly_type in anomaly_types}
        z_valid_aug_t = {anomaly_type: {config: {c: emb.normalize(z_aug) for c, z_aug in z_augs.items()}
                                        for config, z_augs in z_valid_aug[anomaly_type].items()}
                         for anomaly_type in anomaly_types}

        total_loss, f1score = dict(), dict()
        W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
        for anomaly_type in anomaly_types:
            total_loss[anomaly_type] = dict()
            f1score[anomaly_type] = dict()
            for config in z_train_aug_t[anomaly_type].keys():
                total_loss[anomaly_type][config] = dict()
                f1score[anomaly_type][config] = dict()
                for train_config in z_train_aug_t[anomaly_type][config].keys():
                    X = torch.cat([z_train_t, z_train_aug_t[anomaly_type][config][train_config]], dim=0).to(
                        args.device)
                    y = torch.tensor(np.concatenate([
                        np.zeros((len(train_inlier_index), x_train_np.shape[1])),
                        train_labels[anomaly_type][config][train_config]], axis=0)).to(args.device)
                    # X = torch.cat([z_train_aug_t[anomaly_type][config][train_config]], dim=0).to(
                    #     args.device)
                    # y = torch.tensor(np.concatenate([
                    #     train_labels[anomaly_type][config][train_config]], axis=0)).to(args.device)
                    classify_model = None
                    total_loss[anomaly_type][config][train_config] = dict()
                    f1score[anomaly_type][config][train_config] = dict()
                    for valid_config in z_train_aug_t[anomaly_type][config].keys():
                        total_loss[anomaly_type][config][train_config][valid_config] = W_loss(
                            z_train_aug_t[anomaly_type][config][train_config],
                            z_valid_aug_t[anomaly_type][config][valid_config]).item()
                        if classify_model == None:
                            classify_model = train_classify_model(args=args, X_train=X, y_train=y)
                        y_pred = classify(model=classify_model,
                                          X_valid=z_valid_aug_t[anomaly_type][config][valid_config].detach())
                        f1score[anomaly_type][config][train_config][valid_config] = f1_score(
                            torch.tensor(np.array(valid_labels[anomaly_type][config][valid_config])).reshape(-1),
                            y_pred.reshape(-1))
                        print(f'{anomaly_type}, {train_config}, {valid_config}, '
                              f'{total_loss[anomaly_type][config][train_config][valid_config]}, '
                              f'{f1score[anomaly_type][config][train_config][valid_config]}')
                        print(f'{anomaly_type}, {train_config}, {valid_config}, '
                              f'{total_loss[anomaly_type][config][train_config][valid_config]}')
                # visualize_anomaly(z_train, z_valid, z_train_aug[anomaly_type][config],
                #                   z_valid_aug[anomaly_type][config], config, args.trail, anomaly_type)

        # X = torch.cat(
        #     [z_train_t, torch.cat(
        #         [torch.cat([torch.cat([z_aug_t for z_aug_t in z_augs_t.values()], dim=0) for z_augs_t in
        #                     z_train_aug_t[anomaly_type].values()], dim=0) for anomaly_type in
        #          anomaly_types], dim=0)], dim=0).detach()
        # y = torch.tensor(np.concatenate([np.zeros((len(train_inlier_index), x_train_np.shape[1])), np.concatenate(
        #     [np.concatenate([np.concatenate([label for label in labels.values()], axis=0)
        #                      for labels in train_labels[anomaly_type].values()], axis=0)
        #      for anomaly_type in anomaly_types], axis=0)], axis=0)).to(args.device)
        # classify_model = train_classify_model(args=args, X_train=X, y_train=y)
        # for config in z_train_aug_t['spike'].keys():
        #     for valid_config in z_train_aug_t['spike'][config].keys():
        #         y_pred = classify(model=classify_model,
        #                           X_valid=z_valid_aug_t['spike'][config][valid_config].detach())
        #         f1 = f1_score(torch.tensor(np.array(valid_labels['spike'][config][valid_config])).reshape(-1),
        #                       y_pred.reshape(-1))
        #         print(f'spike, {config}, {valid_config}, {f1}')
        #
        # valid_anomaly_types_list = [['spike']]
        # valid_point_list = [{'spike_level': 2}]
        # x_valid_augs_list, valid_labels_list = list(), list()
        # for index, valid_anomaly_types in enumerate(valid_anomaly_types_list):
        #     x_valid_augs, valid_labels = list(), list()
        #     for start_idx, anomaly_type in enumerate(valid_anomaly_types):
        #         x_augs, labels = list(), list()
        #         for i in valid_outlier_index[
        #                  start_idx * (len(valid_outlier_index) // len(valid_anomaly_types)):(start_idx + 1) * (len(
        #                      valid_outlier_index) // len(valid_anomaly_types))]:
        #             if anomaly_type == 'platform':
        #                 x_aug, label = inject_platform(x_valid_np[i], valid_point_list[index]['platform_level'],
        #                                                np.random.uniform(0, 0.5),
        #                                                valid_point_list[index]['platform_length'])
        #             elif anomaly_type == 'mean':
        #                 x_aug, label = inject_mean(x_valid_np[i], valid_point_list[index]['mean_level'],
        #                                            np.random.uniform(0, 0.5),
        #                                            valid_point_list[index]['mean_length'])
        #             elif anomaly_type == 'spike':
        #                 x_aug, label = inject_spike(x_valid_np[i], valid_point_list[index]['spike_level'],
        #                                             np.random.uniform(0, 1))
        #             elif anomaly_type == 'amplitude':
        #                 x_aug, label = inject_amplitude(x_valid_np[i], valid_point_list[index]['amplitude_level'],
        #                                                 np.random.uniform(0, 0.5),
        #                                                 valid_point_list[index]['amplitude_length'])
        #             elif anomaly_type == 'trend':
        #                 x_aug, label = inject_trend(x_valid_np[i], valid_point_list[index]['trend_slope'],
        #                                             np.random.uniform(0, 0.5),
        #                                             valid_point_list[index]['trend_length'])
        #             elif anomaly_type == 'variance':
        #                 x_aug, label = inject_variance(x_valid_np[i], valid_point_list[index]['variance_level'],
        #                                                np.random.uniform(0, 0.5),
        #                                                valid_point_list[index]['variance_length'])
        #             else:
        #                 raise Exception('Unsupported anomaly_type.')
        #             x_augs.append(x_aug)
        #             labels.append(label)
        #         x_valid_augs.append(x_augs)
        #         valid_labels.append(labels)
        #     x_valid_augs_list.append(x_valid_augs)
        #     valid_labels_list.append(valid_labels)
        # z_valid_augs = [
        #     model(torch.tensor(np.concatenate(x_augs, axis=0)).float().unsqueeze(1).to(args.device)).detach()
        #     for x_augs in x_valid_augs_list]
        # valid_labels = [(np.concatenate(labels, axis=0)) for labels in valid_labels_list]
        # z_valid_augs_t = [emb.normalize(z_aug) for z_aug in z_valid_augs]
        # total_loss = list()
        # f1score = list()
        # W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
        # for i, z_valid_aug_t in enumerate(z_valid_augs_t):
        #     total_loss.append(W_loss(torch.cat([torch.cat([torch.cat([z_aug_t[:len(z_aug_t) // 10]
        #                                                               for z_aug_t in z_augs_t.values()], dim=0)
        #                                                    for z_augs_t in z_train_aug_t[anomaly_type].values()],
        #                                                   dim=0)
        #                                         for anomaly_type in anomaly_types], dim=0), z_valid_aug_t).item())
        #     y_pred = classify(model=classify_model, X_valid=z_valid_aug_t)
        #     f1score.append(f1_score(torch.tensor(valid_labels[i]).reshape(-1), y_pred.reshape(-1)))
        #     print(f'wd: {total_loss[i]}, f1score: {f1score[i]}')
        return total_loss, f1score
    else:
        x_valid_aug, valid_labels = list(), list()
        for anomaly_type in valid_anomaly_types:
            for i in valid_outlier_index:
                if anomaly_type == 'platform':
                    x_aug, l = inject_platform(x_valid_np[i], valid_point[anomaly_type]['level'],
                                               np.random.uniform(0, 0.5), valid_point[anomaly_type]['length'])
                elif anomaly_type == 'mean':
                    x_aug, l = inject_mean(x_valid_np[i], valid_point[anomaly_type]['level'],
                                           np.random.uniform(0, 0.5), valid_point[anomaly_type]['length'])
                elif anomaly_type == 'spike':
                    x_aug, l = inject_spike(x_valid_np[i], valid_point[anomaly_type]['level'],
                                            np.random.uniform(0, 0.5))
                elif anomaly_type == 'amplitude':
                    x_aug, l = inject_amplitude(x_valid_np[i], valid_point[anomaly_type]['level'],
                                                np.random.uniform(0, 0.5), valid_point[anomaly_type]['length'])
                elif anomaly_type == 'trend':
                    x_aug, l = inject_trend(x_valid_np[i], valid_point[anomaly_type]['slope'],
                                            np.random.uniform(0, 0.5), valid_point[anomaly_type]['length'])
                elif anomaly_type == 'variance':
                    x_aug, l = inject_variance(x_valid_np[i], valid_point[anomaly_type]['level'],
                                               np.random.uniform(0, 0.5), valid_point[anomaly_type]['length'])
                else:
                    raise Exception(f'Unsupported anomaly_type {anomaly_type}.')
                x_valid_aug.append(x_aug)
                valid_labels.append(l)

        train_p = dict()
        for k, v in train_point.items():
            s = k.split('_')
            if train_p.get(s[0]) is None:
                train_p[s[0]] = dict()
            train_p[s[0]][s[1]] = v
        x_train_aug, train_labels = list(), list()
        for anomaly_type in anomaly_types:
            for i in train_outlier_index:
                if anomaly_type == 'platform':
                    x_aug, l = inject_platform(x_train_np[i], train_p[anomaly_type]['level'],
                                               np.random.uniform(0, 0.5), train_p[anomaly_type]['length'])
                elif anomaly_type == 'mean':
                    x_aug, l = inject_mean(x_train_np[i], train_p[anomaly_type]['level'],
                                           np.random.uniform(0, 0.5), train_p[anomaly_type]['length'])
                elif anomaly_type == 'spike':
                    x_aug, l = inject_spike(x_train_np[i], train_p[anomaly_type]['level'], np.random.uniform(0, 0.5))
                elif anomaly_type == 'amplitude':
                    x_aug, l = inject_amplitude(x_train_np[i], train_p[anomaly_type]['level'],
                                                np.random.uniform(0, 0.5), train_p[anomaly_type]['length'])
                elif anomaly_type == 'trend':
                    x_aug, l = inject_trend(x_train_np[i], train_p[anomaly_type]['slope'],
                                            np.random.uniform(0, 0.5), train_p[anomaly_type]['length'])
                elif anomaly_type == 'variance':
                    x_aug, l = inject_variance(x_train_np[i], train_p[anomaly_type]['level'],
                                               np.random.uniform(0, 0.5), train_p[anomaly_type]['length'])
                else:
                    raise Exception(f'Unsupported anomaly_type: {anomaly_type}.')
                x_train_aug.append(x_aug)
                train_labels.append(l)
        z_train_aug = model(torch.tensor(np.array(x_train_aug)).float().unsqueeze(1).to(args.device)).detach()
        z_valid_aug = model(torch.tensor(np.array(x_valid_aug)).float().unsqueeze(1).to(args.device)).detach()
        z_train_t, z_valid_t, _ = emb(z_train[train_inlier_index].clone().squeeze(),
                                      z_valid[valid_inlier_index].clone().squeeze(),
                                      torch.cat([z_train_aug, z_valid_aug], dim=0))
        z_train_aug_t = emb.normalize(emb=z_train_aug)
        z_valid_aug_t = emb.normalize(emb=z_valid_aug)

        if best is False:
            W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
            loss = W_loss(z_train_aug_t, z_valid_aug_t).item()
            # X = torch.cat([z_train_t, z_train_aug_t.detach()], dim=0)
            # y = torch.tensor(np.concatenate([np.zeros((len(train_inlier_index), x_train_np.shape[1])),
            #                                  train_labels], axis=0)).to(args.device)
            # classify_model = train_classify_model(args=args, X_train=X, y_train=y)
            # y_pred = classify(model=classify_model, X_valid=z_valid_aug_t.detach())
            # f1score = f1_score(torch.tensor(np.array(valid_labels)).reshape(-1), y_pred.reshape(-1))
            # return loss, f1score

            # W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
            # loss_tensor = W_loss(z_train_aug_t, z_valid_aug_t)
            # loss = loss_tensor.item()
            f1score_val = 0.0  # Default F1 score
            if calculate_f1:
                X_clf_train = torch.cat([z_train_t, z_train_aug_t], dim=0).detach()
                train_labels_np = np.array(train_labels)
                valid_labels_np = np.array(valid_labels)
                if train_labels_np.size > 0 and train_labels_np.shape[1] == sequence_length and \
                        valid_labels_np.size > 0 and valid_labels_np.shape[1] == sequence_length:
                    y_clf_train_np = np.concatenate([np.zeros((len(train_inlier_index), sequence_length)),
                                                     train_labels_np], axis=0)
                    y_clf_train = torch.tensor(y_clf_train_np, dtype=torch.float32).to(args.device)
                    classify_model, ls = train_classify_model(args=args, X_train=X_clf_train, y_train=y_clf_train,
                                                              sequence_length=sequence_length)
                    y_pred = classify(model=classify_model, X_valid=z_valid_aug_t.detach())
                    if y_pred.shape == valid_labels_np.shape:
                        f1score_val = f1_score(valid_labels_np.reshape(-1), y_pred.reshape(-1), zero_division=0)
                    else:
                        logging.warning(
                            f"F1 Calc: Shape mismatch between valid labels {valid_labels_np.shape} and predictions "
                            f"{y_pred.shape}. F1=0.")
                else:
                    logging.warning(
                        f"F1 Calc: Problem with label shapes/sizes. Train: {train_labels_np.shape}, "
                        f"Valid: {valid_labels_np.shape}, Expected SeqLen: {sequence_length}. F1=0.")
            return loss, f1score_val
        else:
            log_dir = f'logs/training/{args.trail}'
            os.makedirs(log_dir, exist_ok=True)
            visualize(z_train_t.cpu(), z_valid_t.cpu(), z_train_aug_t.cpu(), z_valid_aug_t.cpu())
            return None, None  # Match return signature


def visualize(train, test, train_aug, test_outlier):
    train_np = train.cpu().numpy() if isinstance(train, torch.Tensor) else train
    test_np = test.cpu().numpy() if isinstance(test, torch.Tensor) else test
    train_aug_np = train_aug.cpu().numpy() if isinstance(train_aug, torch.Tensor) else train_aug
    test_outlier_np = test_outlier.cpu().numpy() if isinstance(test_outlier, torch.Tensor) else test_outlier

    all_data_list = []
    if train_np.size > 0: all_data_list.append(train_np)
    if test_np.size > 0: all_data_list.append(test_np)
    if train_aug_np.size > 0: all_data_list.append(train_aug_np)
    if test_outlier_np.size > 0: all_data_list.append(test_outlier_np)

    if not all_data_list:
        logging.warning("Visualize: No data provided for t-SNE.")
        return

    all_data = np.concatenate(all_data_list, axis=0)

    if all_data.shape[0] <= 1:  # TSNE needs more than 1 sample
        logging.warning(f"Visualize: Not enough samples ({all_data.shape[0]}) for t-SNE.")
        return

    xt = TSNE(n_components=2, random_state=42, perplexity=min(30, all_data.shape[0] - 1)).fit_transform(
        all_data)  # Adjust perplexity

    plt.figure(figsize=(8, 6))
    start_idx = 0
    if train_np.size > 0:
        plt.scatter(xt[start_idx:start_idx + len(train_np), 0], xt[start_idx:start_idx + len(train_np), 1], c='b',
                    alpha=0.5, label='Train Inlier')
        start_idx += len(train_np)
    if test_np.size > 0:
        plt.scatter(xt[start_idx:start_idx + len(test_np), 0], xt[start_idx:start_idx + len(test_np), 1], c='g',
                    alpha=0.5, label='Valid Inlier')
        start_idx += len(test_np)
    if train_aug_np.size > 0:
        plt.scatter(xt[start_idx:start_idx + len(train_aug_np), 0], xt[start_idx:start_idx + len(train_aug_np), 1],
                    c='orange', alpha=0.5, label='Train Aug (BO Point)')
        start_idx += len(train_aug_np)
    if test_outlier_np.size > 0:
        plt.scatter(xt[start_idx:start_idx + len(test_outlier_np), 0],
                    xt[start_idx:start_idx + len(test_outlier_np), 1], c='r', alpha=0.5,
                    label='Valid Aug (Target Point)')

    plt.legend()
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.title('t-SNE of Embeddings (Final Best Point vs Target)')
    log_dir = 'logs/viz'  # Example placeholder
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(f'{log_dir}/tsne_visualization_best.pdf', dpi=300, bbox_inches='tight')
    # plt.show() # Avoid showing plots in non-interactive environments
    plt.close()


def visualize_anomaly(train_inlier, valid_inlier, train_augs, valid_augs, config_name, trail, anomaly_type):
    train_aug = torch.cat(list(train_augs.values()), dim=0)
    valid_aug = torch.cat(list(valid_augs.values()), dim=0)
    aug = torch.cat([train_aug, valid_aug], dim=0)
    all_data = torch.cat([train_inlier, valid_inlier, aug], dim=0).to('cpu').numpy()
    start_idx = 0

    xt = TSNE(n_components=2, random_state=42).fit_transform(all_data)
    plt.figure(figsize=(8, 6))
    plt.scatter(xt[:len(train_inlier), 0], xt[:len(train_inlier), 1], c='b', alpha=0.5)
    start_idx += len(train_inlier)
    plt.scatter(xt[start_idx:start_idx + len(valid_inlier), 0], xt[start_idx:start_idx + len(valid_inlier), 1], c='g',
                alpha=0.5)
    start_idx += len(valid_inlier)
    legend_elements = list()
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label='Train inliers', markerfacecolor='b', markersize=10))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label='Test inliers', markerfacecolor='g', markersize=10))

    # train outliers
    cmap = cm.get_cmap('Greys')
    configs = np.array(list(train_augs.keys()))
    normalized_values = np.arange(0.2, 1, 0.8 / len(configs))
    colors = [cmap(val) for val in normalized_values]
    for i, value in enumerate(configs):
        plt.scatter(xt[start_idx:start_idx + len(train_augs[value]), 0],
                    xt[start_idx:start_idx + len(train_augs[value]), 1],
                    c=[colors[i]] * (len(train_augs[value])), alpha=0.5)
        start_idx += len(train_augs[value])
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=f'Train outliers ({config_name}={value})',
                   markerfacecolor=colors[i], markersize=10))

    # test outliers
    cmap = cm.get_cmap('Reds')
    colors = [cmap(val) for val in normalized_values]
    for i, value in enumerate(configs):
        plt.scatter(xt[start_idx:start_idx + len(valid_augs[value]), 0],
                    xt[start_idx:start_idx + len(valid_augs[value]), 1],
                    c=[colors[i]] * (len(valid_augs[value])), alpha=0.5)
        start_idx += len(valid_augs[value])
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=f'Test outliers ({config_name}={value})',
                   markerfacecolor=colors[i], markersize=10))

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f'{anomaly_type} {config_name}')
    plt.tight_layout()
    plt.savefig(f'logs/training/{trail}/{anomaly_type}_{config_name}.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    num_columns = 2  # Number of columns for the legend
    plt.figure(figsize=(12, len(legend_elements) * 0.2 / num_columns))
    plt.axis('off')
    plt.legend(handles=legend_elements, loc='center', ncol=num_columns, fontsize=8)
    plt.savefig(f'logs/training/{trail}/{anomaly_type}_{config_name}_legend.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


def visualize_time_series(time_series):
    plt.figure(figsize=(10, 4))  # Smaller figure
    plt.plot(time_series, label="Time Series", color="blue", linewidth=1)
    plt.title("Time Series Visualization", fontsize=14)
    plt.xlabel("Time Step", fontsize=10)
    plt.ylabel("Value", fontsize=10)
    plt.grid(alpha=0.3)
    plt.show()
    plt.close()

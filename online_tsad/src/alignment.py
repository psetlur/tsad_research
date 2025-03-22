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
from scipy.stats import truncnorm, bernoulli
from matplotlib.lines import Line2D
import matplotlib.cm as cm

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
    return ts_row, label


def inject_spike(ts_row, p):
    direction = np.random.rand(len(ts_row)) < 0.5
    spikes = np.zeros(len(ts_row))
    spikes[direction] = truncnorm.rvs(-np.inf, -2, loc=0, scale=1, size=direction.sum())
    spikes[~direction] = truncnorm.rvs(2, np.inf, loc=0, scale=1, size=(~direction).sum())
    mask = bernoulli.rvs(p=p, size=len(ts_row))
    ts_row = np.array(ts_row)
    ts_row += mask * spikes
    label = mask * np.ones(len(ts_row))
    return ts_row, label


def inject_spike_with_one(ts_row, start):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    if np.random.rand() < 0.5:
        left, right = 2, np.inf
    else:
        left, right = -np.inf, -2
    start_a = int(len(ts_row) * start)
    ts_row[start_a] += truncnorm.rvs(left, right, loc=0, scale=1)
    label[start_a] = 1
    return ts_row, label


def inject(anomaly_type, ts, trail, config):
    if anomaly_type == 'platform':
        return inject_platform(ts, *config)
    elif anomaly_type == 'mean':
        return inject_mean(ts, *config)
    elif anomaly_type == 'spike':
        if trail == 'inject_spike':
            return inject_spike(ts, *config)
        else:
            return inject_spike_with_one(ts, *config)
    else:
        raise Exception('Unsupported anomaly_type.')


def train_classify_model(args, X_train, y_train):
    model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 512), ).to(args.device)
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


def black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point=None,
                       valid_anomaly_types=None, train_point=None, best=False):
    ratio_anomaly = 0.1
    anomaly_types = ['platform', 'mean']

    with torch.no_grad():
        z_train, x_train_np = [], []
        for x_batch in train_dataloader:
            c_x = model(x_batch.to(args.device)).detach()
            z_train.append(c_x)
            x_train_np.append(x_batch.numpy())
        z_train = torch.cat(z_train, dim=0)
        x_train_np = np.concatenate(x_train_np, axis=0).reshape(len(z_train), -1)

        z_valid, x_valid_np = [], []
        for x_batch in val_dataloader:
            c_x = model(x_batch.to(args.device)).detach()
            z_valid.append(c_x)
            x_valid_np.append(x_batch.numpy())
        z_valid = torch.cat(z_valid, dim=0)
        x_valid_np = np.concatenate(x_valid_np, axis=0).reshape(len(z_valid), -1)

        train_inlier_index, train_outlier_index = train_test_split(range(len(x_train_np)),
                                                                   train_size=1 - ratio_anomaly, random_state=0)
        valid_inlier_index, valid_outlier_index = train_test_split(range(len(x_valid_np)),
                                                                   train_size=1 - ratio_anomaly, random_state=0)

        if valid_point == None:
            emb = EmbNormalizer()

            if args.trail == 'inject_spike_with_one':
                x_train_aug, train_labels = list(), list()
                for i in train_outlier_index:
                    x = x_train_np[i]
                    xa, l = inject('spike', x, trail, [np.random.uniform(0, 0.5)])
                    x_train_aug.append(xa)
                    train_labels.append(l)

                x_valid_aug, valid_labels = list(), list()
                for i in valid_outlier_index:
                    x = x_valid_np[i]
                    xa, l = inject('spike', x, trail, [np.random.uniform(0, 0.5)])
                    x_valid_aug.append(xa)
                    valid_labels.append(l)

                z_train_aug = model(torch.tensor(np.array(x_train_aug)).float().unsqueeze(1).to(args.device)).detach()
                z_valid_aug = model(torch.tensor(np.array(x_valid_aug)).float().unsqueeze(1).to(args.device)).detach()
                z_train_t, z_valid_t, _ = emb(z_train[train_inlier_index].clone().squeeze(),
                                              z_valid[valid_inlier_index].clone().squeeze(),
                                              z_train_aug.squeeze())
                z_train_aug_t = emb.normalize(z_train_aug)
                z_valid_aug_t = emb.normalize(z_valid_aug)

                W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
                wd = W_loss(z_train_aug_t, z_valid_aug_t).item()
                X = torch.cat([z_train_t, z_train_aug_t.detach()], dim=0)
                y = torch.tensor(np.concatenate([np.zeros((len(train_inlier_index), x_train_np.shape[1])),
                                                 train_labels], axis=0)).to(args.device)
                classify_model = train_classify_model(args=args, X_train=X, y_train=y)
                y_pred = classify(model=classify_model, X_valid=z_valid_aug_t.detach())
                f1score = f1_score(torch.tensor(np.array(valid_labels)).reshape(-1), y_pred.reshape(-1))
                print(f'wd: {wd}, f1score: {f1score}')
                visualize(z_train.detach(), z_train_aug.detach(), z_valid.detach(), z_valid_aug.detach())
            else:
                train_p = np.round(np.arange(0.1, 1.1, 0.1), 1)
                valid_p = np.round(np.arange(0.1, 1.1, 0.1), 1)
                x_train_aug_dict, train_labels_dict = dict(), dict()
                for p in train_p:
                    x_train_aug, train_labels = list(), list()
                    for i in train_outlier_index:
                        x = x_train_np[i]
                        xa, l = inject('spike', x, trail, [p])
                        x_train_aug.append(xa)
                        train_labels.append(l)
                    x_train_aug_dict[p] = x_train_aug
                    train_labels_dict[p] = train_labels

                x_valid_aug_dict, valid_labels_dict = dict(), dict()
                for p in valid_p:
                    x_valid_aug, valid_labels = list(), list()
                    for i in valid_outlier_index:
                        x = x_valid_np[i]
                        xa, l = inject('spike', x, trail, [p])
                        x_valid_aug.append(xa)
                        valid_labels.append(l)
                    x_valid_aug_dict[p] = x_valid_aug
                    valid_labels_dict[p] = valid_labels

                z_train_aug = {p: model(torch.tensor(np.array(x_aug)).float().unsqueeze(1).to(args.device)).detach()
                               for p, x_aug in x_train_aug_dict.items()}
                z_valid_aug = {p: model(torch.tensor(np.array(x_aug)).float().unsqueeze(1).to(args.device)).detach()
                               for p, x_aug in x_valid_aug_dict.items()}
                z_train_t, z_valid_t, _ = emb(z_train[train_inlier_index].clone().squeeze(),
                                              z_valid[valid_inlier_index].clone().squeeze(),
                                              torch.cat(list(z_train_aug.values())).squeeze())
                z_train_aug_t = {p: emb.normalize(z_aug) for p, z_aug in z_train_aug.items()}
                z_valid_aug_t = {p: emb.normalize(z_aug) for p, z_aug in z_valid_aug.items()}

                W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
                wd, f1score = dict(), dict()
                for tp in train_p:
                    wd[tp], f1score[tp] = dict(), dict()
                    classify_model = None
                    X = torch.cat([z_train_t, z_train_aug_t[tp].detach()], dim=0)
                    y = torch.tensor(np.concatenate([np.zeros((len(train_inlier_index), x_train_np.shape[1])),
                                                     train_labels_dict[tp]], axis=0)).to(args.device)
                    for vp in valid_p:
                        wd[tp][vp] = W_loss(z_train_aug_t[tp], z_valid_aug_t[vp]).item()
                        if classify_model == None:
                            classify_model = train_classify_model(args=args, X_train=X, y_train=y)
                        y_pred = classify(model=classify_model, X_valid=z_valid_aug_t[vp].detach())
                        f1score[tp][vp] = f1_score(torch.tensor(np.array(valid_labels)).reshape(-1), y_pred.reshape(-1))
                        print(f'train_p: {tp}, valid_p: {vp}, wd: {wd[tp][vp]}, f1score: {f1score[tp][vp]}')
                visualize_multi(z_train, z_valid, z_train_aug, z_valid_aug, train_p, valid_p, 'p', args.trail, 'spike')

            return wd, f1score
        else:
            x_valid_aug, valid_labels = list(), list()
            inlier_num = 0
            for i in valid_outlier_index:
                for anomaly_type in anomaly_types:
                    if anomaly_type not in valid_anomaly_types:
                        x_valid_aug.append(x_valid_np[i])
                        valid_labels.append(np.zeros(len(x_valid_np[i])))
                        inlier_num += 1

            for i in valid_outlier_index:
                for anomaly_type in anomaly_types:
                    if anomaly_type not in valid_anomaly_types:
                        continue
                    else:
                        if anomaly_type == 'spike':
                            x_aug, l = inject(anomaly_type=anomaly_type, ts=x_valid_np[i],
                                              config=[valid_point[anomaly_type]['level'], np.random.uniform(0, 0.5)])
                        else:
                            x_aug, l = inject(anomaly_type=anomaly_type, ts=x_valid_np[i],
                                              config=[valid_point[anomaly_type]['level'], np.random.uniform(0, 0.5),
                                                      valid_point[anomaly_type]['length']])
                        x_valid_aug.append(x_aug)
                        valid_labels.append(l)

            emb = EmbNormalizer()
            z_valid_aug = model(torch.tensor(np.array(x_valid_aug)).float().unsqueeze(1).to(args.device)).detach()
            z_train_t, z_valid_t, z_valid_aug_t = emb(z_train[train_inlier_index].clone().squeeze(),
                                                      z_valid[valid_inlier_index].clone().squeeze(), z_valid_aug)

            train_p = dict()
            for k, v in train_point.items():
                s = k.split('_')
                if train_p.get(s[0]) is None:
                    train_p[s[0]] = dict()
                train_p[s[0]][s[1]] = v
            x_train_aug, train_labels = list(), list()
            for i in train_outlier_index:
                for anomaly_type in anomaly_types:
                    if anomaly_type == 'spike':
                        x_aug, l = inject(anomaly_type=anomaly_type, ts=x_train_np[i],
                                          config=[train_p[anomaly_type]['level'], np.random.uniform(0, 0.5)])
                    else:
                        x_aug, l = inject(anomaly_type=anomaly_type, ts=x_train_np[i],
                                          config=[train_p[anomaly_type]['level'], np.random.uniform(0, 0.5),
                                                  train_p[anomaly_type]['length']])
                    x_train_aug.append(x_aug)
                    train_labels.append(l)
            z_train_aug = model(torch.tensor(np.array(x_train_aug)).float().unsqueeze(1).to(args.device)).detach()
            z_train_aug_t = emb.normalize(emb=z_train_aug)

            if best is False:
                W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
                loss = W_loss(z_train_aug_t, z_valid_aug_t)

                X = torch.cat([z_train_t, z_train_aug_t.detach()], dim=0)
                y = torch.tensor(np.concatenate([np.zeros((len(train_inlier_index), x_train_np.shape[1])),
                                                 train_labels], axis=0)).to(args.device)
                classify_model = train_classify_model(args=args, X_train=X, y_train=y)
                y_pred = classify(model=classify_model, X_valid=z_valid_aug_t.detach())
                f1score = f1_score(torch.tensor(np.array(valid_labels)).reshape(-1), y_pred.reshape(-1))
                return loss.item(), f1score
            else:
                z_valid = torch.cat([z_valid, z_valid_aug[:inlier_num]], dim=0)
                z_valid_aug = z_valid_aug[inlier_num:]
                visualize(z_train.detach(), z_train_aug.detach(), z_valid.detach(), z_valid_aug.detach())


def visualize(train, trn_aug, test, test_aug):
    xt = TSNE(n_components=2, random_state=42).fit_transform(
        torch.cat([train, trn_aug, test, test_aug], dim=0).cpu().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(xt[:len(train), 0], xt[:len(train), 1], c='b', alpha=0.5, label='Train Normal')
    plt.scatter(xt[len(train):len(train) + len(trn_aug), 0], xt[len(train):len(train) + len(trn_aug), 1], c='orange',
                alpha=0.5, label='Train Augmented')
    plt.scatter(xt[len(train) + len(trn_aug):len(train) + len(trn_aug) + len(test), 0],
                xt[len(train) + len(trn_aug):len(train) + len(trn_aug) + len(test), 1], c='g', alpha=0.5,
                label='Test Normal')
    plt.scatter(xt[len(train) + len(trn_aug) + len(test):len(train) + len(trn_aug) + len(test) + len(test_aug), 0],
                xt[len(train) + len(trn_aug) + len(test):len(train) + len(trn_aug) + len(test) + len(test_aug), 1],
                c='r', alpha=0.5, label='Test Anomaly')
    plt.legend()
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.title('Optimized train config vs test config')
    log_dir = f'logs/training/hpo'
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(f'{log_dir}/visualization.pdf', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()


def visualize_multi(train_inlier, valid_inlier, train_augs, valid_augs, train_configs, valid_configs, config_name,
                    trail, anomaly_type):
    aug = torch.cat([torch.cat(list(train_augs.values()), dim=0), torch.cat(list(valid_augs.values()), dim=0)], dim=0)
    all_data = torch.cat([train_inlier, valid_inlier, aug], dim=0).to('cpu').numpy()
    start_size = 0
    legend_elements = list()
    xt = TSNE(n_components=2, random_state=42).fit_transform(all_data)
    plt.figure(figsize=(12, 8))

    # train inliers
    plt.scatter(xt[start_size:start_size + len(train_inlier), 0], xt[start_size:start_size + len(train_inlier), 1],
                c='b', alpha=0.5)
    start_size += len(train_inlier)
    # test inliers
    plt.scatter(xt[start_size:start_size + len(valid_inlier), 0],
                xt[start_size:start_size + len(valid_inlier), 1], c='g', alpha=0.5)
    start_size += len(valid_inlier)
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label='Train inliers', markerfacecolor='b', markersize=10))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label='Test inliers', markerfacecolor='g', markersize=10))

    # train outliers
    cmap = cm.get_cmap('Greys')

    normalized_values = (train_configs - np.min(train_configs)) / (np.max(train_configs) - np.min(train_configs))
    normalized_values = normalized_values * (1 - 0.1) + 0.1
    colors = [cmap(val) for val in normalized_values]
    for i, value in enumerate(train_configs):
        plt.scatter(xt[start_size:start_size + len(train_augs[value]), 0],
                    xt[start_size:start_size + len(train_augs[value]), 1],
                    c=[colors[i]] * len(train_augs[value]), alpha=0.5)
        start_size += len(train_augs[value])
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=f'Train outliers ({config_name}={value})',
                   markerfacecolor=colors[i], markersize=10))

    # test outliers
    cmap = cm.get_cmap('Reds')
    normalized_values = (valid_configs - np.min(valid_configs)) / (np.max(valid_configs) - np.min(valid_configs))
    normalized_values = normalized_values * (1 - 0.1) + 0.1
    colors = [cmap(val) for val in normalized_values]
    for i, value in enumerate(valid_configs):
        plt.scatter(xt[start_size:start_size + len(valid_augs[value]), 0],
                    xt[start_size:start_size + len(valid_augs[value]), 1],
                    c=[colors[i]] * len(valid_augs[value]), alpha=0.5)
        start_size += len(valid_augs[value])
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=f'Test outliers ({config_name}={value})',
                   markerfacecolor=colors[i], markersize=10))

    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f'{anomaly_type}')
    plt.tight_layout()
    plt.savefig(f'logs/training/{trail}/{anomaly_type}.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

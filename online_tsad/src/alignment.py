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
from scipy.stats import bernoulli
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import itertools


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


def inject_spike(ts_row, level, p):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    mask = bernoulli.rvs(p=p, size=len(ts_row)).astype(bool)
    ts_row[mask] *= level
    label[mask] = 1
    return ts_row, label


def inject(anomaly_type, ts, config):
    if anomaly_type == 'platform':
        return inject_platform(ts, *config)
    elif anomaly_type == 'mean':
        return inject_mean(ts, *config)
    elif anomaly_type == 'spike':
        return inject_spike(ts, *config)
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
    min_level = -1.0
    min_length = 0.2
    max_level = 2.0
    max_length = 0.6
    level_step = 2.0
    length_step = 0.3
    fixed_level = 0.5
    fixed_length = 0.3
    min_spike_level = 2
    min_spike_p = 0.01
    max_spike_level = 22
    max_spike_p = 0.06
    spike_level_step = 18
    spike_p_step = 0.04
    fixed_spike_level = 2
    fixed_spike_p = 0.01
    anomaly_types = ['platform', 'mean', 'spike']

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
        emb = EmbNormalizer()
        if valid_point == None:
            x_aug_list = list()
            for anomaly_type in anomaly_types:
                x_augs = list()
                for i in train_outlier_index:
                    if anomaly_type == 'spike':
                        x_aug, _ = inject(anomaly_type=anomaly_type, ts=x_train_np[i],
                                          config=[fixed_spike_level, fixed_spike_p])
                    else:
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
                            x_augs[anomaly_type][(level, round(length, 2))] = list()
                            labels[anomaly_type][(level, round(length, 2))] = list()
                        for i in outlier_index:
                            if anomaly_type != 'spike':
                                x_aug, label = inject(anomaly_type=anomaly_type, ts=x_np[i],
                                                      config=[level, np.random.uniform(0, 0.5), length])
                                x_augs[anomaly_type][(round(level, 1), round(length, 1))].append(np.array(x_aug))
                                labels[anomaly_type][(round(level, 1), round(length, 1))].append(np.array(label))
                            else:
                                x_aug, label = inject(anomaly_type=anomaly_type, ts=x_np[i], config=[level, length])
                                x_augs[anomaly_type][(level, round(length, 2))].append(np.array(x_aug))
                                labels[anomaly_type][(level, round(length, 2))].append(np.array(label))

                    if anomaly_type != 'spike':
                        for level in np.arange(min_level, max_level, level_step):
                            for length in np.arange(min_length, max_length, length_step):
                                arg(level=level, length=length)
                                x_augs[anomaly_type][(round(level, 1), round(length, 1))] = np.array(
                                    x_augs[anomaly_type][(round(level, 1), round(length, 1))])
                                x_augs[anomaly_type][(round(level, 1), round(length, 1))] = torch.tensor(
                                    x_augs[anomaly_type][(round(level, 1), round(length, 1))]).float().unsqueeze(
                                    1).to(
                                    args.device)
                                x_augs[anomaly_type][(round(level, 1), round(length, 1))] = model(
                                    x_augs[anomaly_type][(round(level, 1), round(length, 1))]).detach()
                    else:
                        for level in np.arange(min_spike_level, max_spike_level, spike_level_step):
                            for p in np.arange(min_spike_p, max_spike_p, spike_p_step):
                                arg(level=level, length=p)
                                x_augs[anomaly_type][(level, round(p, 2))] = np.array(
                                    x_augs[anomaly_type][(level, round(p, 2))])
                                x_augs[anomaly_type][(level, round(p, 2))] = torch.tensor(
                                    x_augs[anomaly_type][(level, round(p, 2))]).float().unsqueeze(1).to(args.device)
                                x_augs[anomaly_type][(level, round(p, 2))] = model(
                                    x_augs[anomaly_type][(level, round(p, 2))]).detach()

                combinations = list(itertools.product(*[x_augs[anomaly_type] for anomaly_type in anomaly_types]))
                for combination in combinations:
                    data = (list(), list())
                    for anomaly_type, ac in zip(anomaly_types, combination):
                        data[0].append(x_augs[anomaly_type][ac])
                        data[1].append(labels[anomaly_type][ac])
                    z_augs_dict[combination] = torch.cat(data[0], dim=0)
                    labels_dict[combination] = np.concatenate(data[1], axis=0)
                return z_augs_dict, labels_dict

            z_train_aug, train_labels = argument(x_np=x_train_np, outlier_index=train_outlier_index)
            z_valid_aug, valid_labels = argument(x_np=x_valid_np, outlier_index=valid_outlier_index)

            z_train_aug_t = {config: emb.normalize(z_aug) for config, z_aug in z_train_aug.items()}
            z_valid_aug_t = {config: emb.normalize(z_aug) for config, z_aug in z_valid_aug.items()}

            total_loss = dict()
            f1score = dict()
            W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
            for train_config, z_train_config_aug_t in z_train_aug_t.items():
                total_loss[train_config] = dict()
                f1score[train_config] = dict()

                classify_model = None
                for valid_config, z_valid_config_aug_t in z_valid_aug_t.items():
                    total_loss[train_config][valid_config] = W_loss(z_train_config_aug_t, z_valid_config_aug_t).item()
                    if classify_model is None:
                        X = torch.cat([z_train_t, z_train_config_aug_t], dim=0)
                        y = torch.tensor(np.concatenate([np.zeros((len(train_inlier_index), x_train_np.shape[1])),
                                                         train_labels[train_config]], axis=0)).to(args.device)
                        classify_model = train_classify_model(args=args, X_train=X, y_train=y)

                    y_pred = classify(model=classify_model, X_valid=z_valid_config_aug_t)
                    f1score[train_config][valid_config] = f1_score(torch.tensor(valid_labels[valid_config]).
                                                                   reshape(-1), y_pred.reshape(-1))
                    print(f'train: {train_config}, valid: {valid_config}, '
                          f'wd: {total_loss[train_config][valid_config]}, '
                          f'f1score: {f1score[train_config][valid_config]}')

            # inlier = {'train': z_train_t, 'valid': z_valid_t}
            # outlier = {'train': z_train_aug_t, 'valid': z_valid_aug_t}
            # combinations = list(z_train_aug_t.keys())
            # visualize_multiple(inlier=inlier, outlier=outlier, combinations=combinations, trail=args.trail)
            return total_loss, f1score
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
                                              config=[valid_point[anomaly_type]['level'],
                                                      valid_point[anomaly_type]['p']])
                        else:
                            x_aug, l = inject(anomaly_type=anomaly_type, ts=x_valid_np[i],
                                              config=[valid_point[anomaly_type]['level'], np.random.uniform(0, 0.5),
                                                      valid_point[anomaly_type]['length']])
                        x_valid_aug.append(x_aug)
                        valid_labels.append(l)

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
                                          config=[train_p[anomaly_type]['level'], train_p[anomaly_type]['p']])
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
                visualize(torch.cat([z_train.detach(), z_train_aug.detach()], dim=0),
                          torch.cat([z_valid.detach(), z_valid_aug.detach()], dim=0))

def evaluate_specific_point(args, model, train_dataloader, val_dataloader, test_dataloader, specific_point):
    """
    Evaluate a specific point configuration to get its WD and F1 score
    
    specific_point: Dictionary with keys for the anomaly types present
                   (e.g., {'platform': {'level': 0.5, 'length': 0.3}, 'spike': {'level': 15, 'p': 0.03}})
    """
    # Extract anomaly types from the specific_point dictionary
    valid_anomaly_types = list(specific_point.keys())
    
    # We need to ensure ALL anomaly types are present in train_point
    # even if they're not in valid_anomaly_types
    train_point = {}
    
    # First add default values for all possible anomaly types
    default_values = {
        'platform': {'level': 0.0, 'length': 0.0},
        'mean': {'level': 0.0, 'length': 0.0},
        'spike': {'level': 0.0, 'p': 0.00}
    }
    
    # Add all anomaly types with default values
    for anomaly_type in ['platform', 'mean', 'spike']:
        for param, value in default_values[anomaly_type].items():
            train_point[f"{anomaly_type}_{param}"] = value
    
    # Then override with our specific values
    for anomaly_type in valid_anomaly_types:
        for param, value in specific_point[anomaly_type].items():
            train_point[f"{anomaly_type}_{param}"] = value
    
    # Call black_box_function with the specific point
    wd, f1 = black_box_function(
        args, model, train_dataloader, val_dataloader, test_dataloader,
        valid_point=specific_point,
        valid_anomaly_types=valid_anomaly_types,
        train_point=train_point,
        best=False
    )
    
    return wd, f1


def visualize(train, test):
    xt = TSNE(n_components=2, random_state=42).fit_transform(
        torch.cat([train, test], dim=0).cpu().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(xt[:len(train), 0], xt[:len(train), 1], c='b', alpha=0.5, label='Train Data')
    plt.scatter(xt[len(train):len(train) + len(test), 0], xt[len(train):len(train) + len(test), 1], c='g',
                alpha=0.5, label='Test Data')
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

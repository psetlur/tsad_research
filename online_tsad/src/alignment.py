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


def classify(X_train, y_train, X_test):
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

    y_pred = torch.where(torch.sigmoid(model(X_test).detach()) > 0.5, 1, 0).cpu().numpy()
    return y_pred


def hist_sample(cdf, bins):
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    val = np.random.uniform(bins[bin_idx - 1], bins[bin_idx])
    return val


def black_box_function(model, train_dataloader, val_dataloader, test_dataloader, a_config):
    n_trials = 1
    # ratio_0, ratio_1 = a_config['ratio_0'], a_config['ratio_1']
    # ratio_anomaly = a_config['ratio_anomaly']
    # fixed_level = a_config['fixed_level']
    # fixed_length = a_config['fixed_length']
    # fixed_start = a_config['fixed_start']
    ratio_anomaly = 0.1
    levels = np.round(np.arange(-1.0, 1.1, 0.1), 1)
    lengths = np.round(np.arange(0.2, 0.52, 0.02), 2)
    fixed_level = 0.5
    fixed_length = 0.3
    fixed_start = 0.2

    train_levels = np.round(np.arange(-1.0, 1.2, 0.2), 1)
    train_lengths = np.round(np.arange(0.2, 0.54, 0.04), 2)
    with torch.no_grad():
        z_train, x_train_np = [], []
        for x_batch in train_dataloader:
            c_x = model(x_batch.to(0)).detach().cpu()
            z_train.append(c_x)
            x_train_np.append(x_batch.numpy())
        z_train = torch.cat(z_train, dim=0)
        x_train_np = np.concatenate(x_train_np, axis=0).reshape(len(z_train), -1)
        #
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

        z_test, y_test, t_test = [], [], []
        for x_batch, y_batch in test_dataloader:
            c_x = model(x_batch.to(0)).detach().cpu()
            z_test.append(c_x)
            y_batch_t = np.zeros((x_batch.shape[0], x_batch.shape[2]))
            for i, m in enumerate(y_batch.squeeze()):
                m_start, m_length, _, m_type = m[-4:]
                if m_type != -1:
                    y_batch_t[i, int(m_start):int(m_start) + int(m_length)] = 1
            y_test.append(y_batch_t)
            t_test.append(y_batch[:, 0, -1])
        z_test = torch.cat(z_test, dim=0)
        y_test = np.concatenate(y_test, axis=0)
        t_test = np.concatenate(t_test, axis=0)

        emb = EmbNormalizer()
        total_loss = []
        fscore = []
        for seed in range(n_trials):
            train_index, ttest_index = train_test_split(range(len(x_train_np)), train_size=1 - ratio_anomaly,
                                                        random_state=seed)
            valid_index, vtest_index = train_test_split(range(len(x_valid_np)), train_size=1 - ratio_anomaly,
                                                        random_state=seed)
            # test_index_0, test_index_1 = train_test_split(test_index, train_size=ratio_0/(ratio_0+ratio_1), random_state=seed)

            x_aug_level_list, labels_level_list = [], []
            for level in levels:
                x_aug, labels = [], []
                for i in vtest_index:
                    x = x_valid_np[i]
                    xa, l = inject_platform(x, level, fixed_start, fixed_length)
                    x_aug.append(xa)
                    labels.append(l)
                x_aug_level_list.append(x_aug)
                labels_level_list.append(labels)

            x_aug_length_list, labels_length_list = [], []
            for length in lengths:
                x_aug, labels = [], []
                for i in vtest_index:
                    x = x_valid_np[i]
                    xa, l = inject_platform(x, fixed_level, fixed_start, length)
                    x_aug.append(xa)
                    labels.append(l)
                x_aug_length_list.append(x_aug)
                labels_length_list.append(labels)

            train_x_aug_level_list, train_labels_level_list = [], []
            for level in train_levels:
                x_aug, labels = [], []
                for i in ttest_index:
                    x = x_train_np[i]
                    xa, l = inject_platform(x, level, fixed_start, fixed_length)
                    x_aug.append(xa)
                    labels.append(l)
                train_x_aug_level_list.append(x_aug)
                train_labels_level_list.append(labels)

            train_x_aug_length_list, train_labels_length_list = [], []
            for length in train_lengths:
                x_aug, labels = [], []
                for i in ttest_index:
                    x = x_train_np[i]
                    xa, l = inject_platform(x, fixed_level, fixed_start, length)
                    x_aug.append(xa)
                    labels.append(l)
                train_x_aug_length_list.append(x_aug)
                train_labels_length_list.append(labels)

            x_aug, labels = [], []
            for i in ttest_index:
                x = x_train_np[i]
                # if np.random.random() > 0.5:
                #     xa, l = inject_platform(x, fixed_level_0, fixed_start_0, fixed_length_0)
                # else:
                #     xa, l = inject_platform(x, fixed_level_1, fixed_start_1, fixed_length_1)
                xa, l = inject_platform(x, fixed_level, fixed_start, fixed_length)
                x_aug.append(xa)
                labels.append(l)

            z_aug_level_list = [model(torch.tensor(np.array(x_aug_level)).float().unsqueeze(1).to(0)).detach().cpu()
                                for x_aug_level in x_aug_level_list]
            z_aug_length_list = [model(torch.tensor(np.array(x_aug_length)).float().unsqueeze(1).to(0)).detach().cpu()
                                 for x_aug_length in x_aug_length_list]

            z_train_aug_level_list = [
                model(torch.tensor(np.array(x_aug_level)).float().unsqueeze(1).to(0)).detach().cpu()
                for x_aug_level in train_x_aug_level_list]
            z_train_aug_length_list = [
                model(torch.tensor(np.array(x_aug_length)).float().unsqueeze(1).to(0)).detach().cpu()
                for x_aug_length in train_x_aug_length_list]

            # z_aug = model(torch.tensor(np.array(x_aug)).float().unsqueeze(1).to(0)).detach().cpu()
            # z_train_t, z_aug_t, z_valid_t = emb(z_train[train_index].clone().squeeze(), z_aug.clone().squeeze(),
            #                                     z_valid.clone().squeeze())
            z_aug = model(torch.tensor(np.array(x_aug)).float().unsqueeze(1).to(0)).detach().cpu()
            z_train_t, z_aug_t, z_valid_t = emb(z_train[train_index].clone().squeeze(), z_aug.clone().squeeze(),
                                                z_valid[valid_index].clone().squeeze())
            z_aug_t_level_list = [emb.normalize(z_aug_level) for z_aug_level in z_aug_level_list]
            z_aug_t_length_list = [emb.normalize(z_aug_length) for z_aug_length in z_aug_length_list]
            z_train_aug_t_level_list = [emb.normalize(z_aug_level) for z_aug_level in z_train_aug_level_list]
            z_train_aug_t_length_list = [emb.normalize(z_aug_length) for z_aug_length in z_train_aug_length_list]

            # W_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
            # loss = -W_loss(torch.cat([z_train_t, z_aug_t], dim=0), z_valid_t).item()
            # total_loss.append(loss)

            W_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
            loss = -W_loss(torch.cat(
                [z_train_t, torch.cat(z_train_aug_t_level_list, dim=0), torch.cat(z_train_aug_t_length_list, dim=0)],
                dim=0), torch.cat(
                [z_valid_t, torch.cat(z_aug_t_level_list, dim=0), torch.cat(z_aug_t_length_list, dim=0)], dim=0)).item()
            total_loss.append(loss)

            # z_test_t = emb.normalize(z_test)
            # X = np.concatenate([z_train_t.numpy(), z_aug_t.numpy()], axis=0)
            # y = np.concatenate([np.zeros((len(train_index), x_train_np.shape[1])), labels], axis=0)
            # y_pred = classify(torch.tensor(X).float().to('cuda:0'), torch.tensor(y).float().to('cuda:0'),
            #                   z_test_t.to('cuda:0'))
            # fscore.append(f1_score(y_test.reshape(-1), y_pred.reshape(-1)))

            X = np.concatenate([z_train_t.numpy(), torch.cat(z_train_aug_t_level_list, dim=0).numpy(),
                                torch.cat(z_train_aug_t_length_list, dim=0).numpy()], axis=0)
            y = np.concatenate(
                [np.zeros((len(train_index), x_train_np.shape[1])), np.concatenate(train_labels_level_list, axis=0),
                 np.concatenate(train_labels_length_list, axis=0)], axis=0)
            y_pred = classify(torch.tensor(X).float().to('cuda:0'), torch.tensor(y).float().to('cuda:0'),
                              torch.tensor(np.concatenate([z_valid_t, np.concatenate(z_aug_t_level_list, axis=0),
                                                           np.concatenate(z_aug_t_length_list, axis=0)],
                                                          axis=0)).float().to('cuda:0'))
            y_valid = np.concatenate(
                [np.zeros((len(valid_index), x_valid_np.shape[1])), np.concatenate(labels_level_list, axis=0),
                 np.concatenate(labels_length_list, axis=0)], axis=0)
            fscore.append(f1_score(y_valid.reshape(-1), y_pred.reshape(-1)))

            visualize_fixed_grid(z_train, z_valid, z_train_aug_level_list, z_aug_level_list, train_levels, levels,
                                 'level', f'length{fixed_length}', 1)
            visualize_fixed_grid(z_train, z_valid, z_train_aug_level_list, z_aug_level_list, train_levels, levels,
                                 'level', f'length{fixed_length}', -1)
            visualize_fixed_grid(z_train, z_valid, z_train_aug_length_list, z_aug_length_list, train_lengths, lengths,
                                 'length', f'level{fixed_level}', 1)
            visualize_fixed_grid(z_train, z_valid, z_train_aug_length_list, z_aug_length_list, train_lengths, lengths,
                                 'length', f'level{fixed_level}', -1)

        # total_loss = np.mean(total_loss)
        # fscore = np.mean(fscore)
        #
        # visualize_embedding(z_train, z_aug, z_test, y_test)

    return total_loss, fscore


def visualize_embedding(z_train, z_aug, z_test, y_test):
    y_test_t = np.max(y_test, axis=1)
    xt = TSNE(n_components=2, random_state=42).fit_transform(torch.cat([z_train, z_aug, z_test], dim=0).cpu().numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(xt[:len(z_train), 0], xt[:len(z_train), 1], c='b', alpha=0.5, label='Train Normal')
    plt.scatter(xt[len(z_train):len(z_train) + len(z_aug), 0],
                xt[len(z_train):len(z_train) + len(z_aug), 1], c='orange', alpha=0.5, label='Train Augmented')
    plt.scatter(xt[np.where(y_test_t == 0)[0] + len(z_train) + len(z_aug), 0],
                xt[np.where(y_test_t == 0)[0] + len(z_train) + len(z_aug), 1], c='g', alpha=0.5, label='Test Normal')
    plt.scatter(xt[np.where(y_test_t == 1)[0] + len(z_train) + len(z_aug), 0],
                xt[np.where(y_test_t == 1)[0] + len(z_train) + len(z_aug), 1], c='r', alpha=0.5,
                label='Test Anomaly (Platform)')

    plt.legend()
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig('embedding_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_fixed_config(train, train_aug, test, test_aug, fixed_level, fixed_length, varying_level, varying_length):
    xt = TSNE(n_components=2, random_state=42).fit_transform(
        torch.cat([train, train_aug, test, test_aug], dim=0).cpu().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(xt[:len(train), 0], xt[:len(train), 1], c='b', alpha=0.5, label='Train Normal')
    plt.scatter(xt[len(train):len(train) + len(train_aug), 0], xt[len(train):len(train) + len(train_aug), 1],
                c='orange', alpha=0.5, label='Train Augmented')
    plt.scatter(xt[len(train) + len(train_aug):len(train) + len(train_aug) + len(test), 0],
                xt[len(train) + len(train_aug):len(train) + len(train_aug) + len(test), 1], c='g', alpha=0.5,
                label='Test Normal')
    plt.scatter(xt[len(train) + len(train_aug) + len(test):len(train) + len(train_aug) + len(test) + len(test_aug), 0],
                xt[len(train) + len(train_aug) + len(test):len(train) + len(train_aug) + len(test) + len(test_aug), 1],
                c='r', alpha=0.5, label='Test Anomaly (Platform)')
    plt.legend()
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.title('t-SNE Visualization of Embeddings')
    plt.savefig(f'logs/training/level{fixed_level}length{"{:.2f}".format(fixed_length)}/'
                f'level{varying_level}length{varying_length}_embedding_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_fixed_grid(train_normal, test_normal, train_aug_list, aug_list,
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
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=f'Anomalies (Platform) ({para_name}={value})',
                   markerfacecolor=colors[i], markersize=10)
        )

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
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=f'Anomalies (Platform) ({para_name}={value})',
                   markerfacecolor=colors[i], markersize=10)
        )
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization of Embeddings')
    plt.tight_layout()
    plt.savefig(f'logs/training/fixed_grid/{fixed_value}_{converse}_embedding_visualization.pdf', dpi=300,
                bbox_inches='tight')
    plt.close()

# def visualize_fixed_grid(train_normal, test_normal, aug_list, test_values, para_name, fixed_value, converse=1):
#     aug = torch.cat(aug_list, dim=0)
#     all_data = torch.cat([train_normal, test_normal, aug], dim=0).cpu().numpy()
#     xt = TSNE(n_components=2, random_state=42).fit_transform(all_data)
#     plt.figure(figsize=(10, 6))
#     # train normal
#     plt.scatter(xt[:len(train_normal), 0], xt[:len(train_normal), 1], c='b', alpha=0.5, label='Train Normal')
#     # test normal
#     plt.scatter(xt[len(train_normal):len(train_normal) + len(test_normal), 0],
#                 xt[len(train_normal):len(train_normal) + len(test_normal), 1],
#                 c='g', alpha=0.5, label='Test Normal')
#
#     # train anomalies
#     cmap = cm.get_cmap('Greys')
#     normalized_values = (test_values - np.min(test_values)) / (np.max(test_values) - np.min(test_values))
#     colors = [cmap(val) for val in normalized_values]
#     for i, value in enumerate(test_values):
#         start_idx = len(train_normal) + len(test_normal) + i * len(aug_list[i]) // len(test_values)
#         end_idx = start_idx + len(aug_list[i]) // len(test_values)
#         if converse == 1:
#             plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                         c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
#                         label=f'Anomalies (Platform) ({para_name}={value})')
#         else:
#             plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                         c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
#                         zorder=len(test_values) - i,
#                         label=f'Anomalies (Platform) ({para_name}={value})')
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='Train Normal', markerfacecolor='b', markersize=10),
#         Line2D([0], [0], marker='o', color='w', label='Test Normal', markerfacecolor='g', markersize=10),
#     ]
#     for i, value in enumerate(test_values):
#         legend_elements.append(
#             Line2D([0], [0], marker='o', color='w', label=f'Anomalies (Platform) ({para_name}={value})',
#                    markerfacecolor=colors[i], markersize=10)
#         )
#
#     # test anomalies
#     cmap = cm.get_cmap('Reds')
#     normalized_values = (test_values - np.min(test_values)) / (np.max(test_values) - np.min(test_values))
#     colors = [cmap(val) for val in normalized_values]
#     for i, value in enumerate(test_values):
#         start_idx = len(train_normal) + len(test_normal) + i * len(aug_list[i]) // len(test_values)
#         end_idx = start_idx + len(aug_list[i]) // len(test_values)
#         if converse == 1:
#             plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                         c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
#                         label=f'Anomalies (Platform) ({para_name}={value})')
#         else:
#             plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                         c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
#                         zorder=len(test_values) - i,
#                         label=f'Anomalies (Platform) ({para_name}={value})')
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='Train Normal', markerfacecolor='b', markersize=10),
#         Line2D([0], [0], marker='o', color='w', label='Test Normal', markerfacecolor='g', markersize=10),
#     ]
#     for i, value in enumerate(test_values):
#         legend_elements.append(
#             Line2D([0], [0], marker='o', color='w', label=f'Anomalies (Platform) ({para_name}={value})',
#                    markerfacecolor=colors[i], markersize=10)
#         )
#     plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
#     plt.xlabel('t-SNE 1')
#     plt.ylabel('t-SNE 2')
#     plt.title('t-SNE Visualization of Embeddings')
#     plt.tight_layout()
#     plt.savefig(f'logs/training/fixed_grid/{fixed_value}_{converse}_embedding_visualization.pdf', dpi=300,
#                 bbox_inches='tight')
#     plt.close()

# def visualize_fixed_grid(train, trn_aug, test, test_aug, fixed_level, fixed_length, varying_param_name, varying_values):
#     all_data = torch.cat([train, trn_aug, test, test_aug], dim=0).cpu().numpy()
#     xt = TSNE(n_components=2, random_state=42).fit_transform(all_data)
#     plt.figure(figsize=(8, 6))
#
#     # train normal
#     plt.scatter(xt[:len(train), 0], xt[:len(train), 1], c='b', alpha=0.5, label='Train Normal')
#
#     # train Augmented
#     plt.scatter(xt[len(train):len(train) + len(trn_aug), 0], xt[len(train):len(train) + len(trn_aug), 1], c='orange',
#                 alpha=0.5, label='Train Augmented')
#
#     # test normal
#     plt.scatter(xt[len(train) + len(trn_aug):len(train) + len(trn_aug) + len(test), 0],
#                 xt[len(train) + len(trn_aug):len(train) + len(trn_aug) + len(test), 1], c='g', alpha=0.5,
#                 label='Test Normal')
#
#     # test anomalies
#     cmap = cm.get_cmap('coolwarm', len(varying_values))
#     colors = [cmap(i) for i in range(len(varying_values))]
#
#     for i, value in enumerate(varying_values):
#         start_idx = len(train) + len(trn_aug) + len(test) + i * len(test_aug) // len(varying_values)
#         end_idx = start_idx + len(test_aug) // len(varying_values)
#
#         plt.scatter(xt[start_idx:end_idx, 0], xt[start_idx:end_idx, 1],
#                     c=[colors[i]] * (end_idx - start_idx), alpha=0.5,
#                     label=f'Test Anomaly (Platform) ({varying_param_name}={value:.2f})')
#
#     plt.legend()
#     plt.xlabel('t-SNE 1')
#     plt.ylabel('t-SNE 2')
#     plt.title('t-SNE Visualization of Embeddings')
#     plt.tight_layout()
#
#     plt.savefig(
#         f'logs/training/level{fixed_level}length{"{:.2f}".format(fixed_length)}/{varying_param_name}_embedding_visualization.pdf',
#         dpi=300, bbox_inches='tight'
#     )
#     plt.close()

import os
import numpy as np
import math

import torch
from torch import nn
import geomloss

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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
    ts_row[start_a : start_a + length_a] = level
    label[start_a : start_a + length_a] = 1
    return ts_row, label


def inject_mean(ts_row, level, start, length):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    start_a = int(len(ts_row) * start)
    length_a = int(len(ts_row) * length)
    ts_row[start_a : start_a + length_a] += float(level)
    label[start_a : start_a + length_a] = 1
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
    val = np.random.uniform(bins[bin_idx-1], bins[bin_idx])
    return val


def black_box_function(model, train_dataloader, val_dataloader, test_dataloader, a_config):
    n_trials = 1
    ratio_0, ratio_1 = a_config['ratio_0'], a_config['ratio_1']

    level_0_cdf  = [0, a_config['level_0_h0'] , a_config['level_0_h0']  + a_config['level_0_h1'] , 1]
    length_0_cdf = [0, a_config['length_0_h0'], a_config['length_0_h0'] + a_config['length_0_h1'], 1]
    level_1_cdf  = [0, a_config['level_1_h0'] , a_config['level_1_h0']  + a_config['level_1_h1'] , 1]
    length_1_cdf = [0, a_config['length_1_h0'], a_config['length_1_h0'] + a_config['length_1_h1'], 1]
    bins_level = [-1, -0.33, 0.33, 1]
    bins_length = [0.2, 0.3, 0.4, 0.5]

    with torch.no_grad():
        z_train, x_train_np = [], []
        for x_batch in train_dataloader:
            c_x = model(x_batch.to(0)).detach().cpu()
            z_train.append(c_x)
            x_train_np.append(x_batch.numpy())
        z_train = torch.cat(z_train, dim=0)
        x_train_np = np.concatenate(x_train_np, axis=0).reshape(len(z_train), -1)

        z_valid = []
        for x_batch in val_dataloader:
            c_x = model(x_batch.to(0)).detach().cpu()
            z_valid.append(c_x)
        z_valid = torch.cat(z_valid, dim=0)

        z_test, y_test, t_test = [], [], []
        for x_batch, y_batch in test_dataloader:
            c_x = model(x_batch.to(0)).detach().cpu()
            z_test.append(c_x)
            y_batch_t = np.zeros((x_batch.shape[0], x_batch.shape[2]))
            for i, m in enumerate(y_batch.squeeze()):
                m_start, m_length, _, m_type = m[-4:]
                if m_type != -1:
                    y_batch_t[i, int(m_start):int(m_start)+int(m_length)] = 1
            y_test.append(y_batch_t)
            t_test.append(y_batch[:, 0, -1])
        z_test = torch.cat(z_test, dim=0)
        y_test = np.concatenate(y_test, axis=0)
        t_test = np.concatenate(t_test, axis=0)

        emb = EmbNormalizer()
        total_loss = []
        fscore = []
        for seed in range(n_trials):
            train_index, test_index = train_test_split(range(len(x_train_np)), train_size=1-ratio_0-ratio_1, random_state=seed)
            test_index_0, test_index_1 = train_test_split(test_index, train_size=ratio_0/(ratio_0+ratio_1), random_state=seed)

            x_aug, labels = [], []
            for i in test_index_0:
                x = x_train_np[i]
                xa, l = inject_platform(x, hist_sample(level_0_cdf, bins_level), np.random.uniform(0, 0.5), hist_sample(length_0_cdf, bins_length))
                x_aug.append(xa)
                labels.append(l)
            for i in test_index_1:
                x = x_train_np[i]
                xa, l = inject_mean(x, hist_sample(level_1_cdf, bins_level), np.random.uniform(0, 0.5), hist_sample(length_1_cdf, bins_length))
                x_aug.append(xa)
                labels.append(l)

            z_aug = model(torch.tensor(np.array(x_aug)).float().unsqueeze(1).to(0)).detach().cpu()
            z_train_t, z_aug_t, z_valid_t = emb(z_train[train_index].clone().squeeze(), z_aug.clone().squeeze(), z_valid.clone().squeeze())

            W_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
            loss = -W_loss(torch.cat([z_train_t, z_aug_t], dim=0), z_valid_t).item()
            total_loss.append(loss)

            z_test_t = emb.normalize(z_test)
            X = np.concatenate([z_train_t.numpy(), z_aug_t.numpy()], axis=0)
            y = np.concatenate([np.zeros((len(train_index), x_train_np.shape[1])), labels], axis=0)
            y_pred = classify(torch.tensor(X).float().to('cuda:0'), torch.tensor(y).float().to('cuda:0'), z_test_t.to('cuda:0'))
            fscore.append(f1_score(y_test.reshape(-1), y_pred.reshape(-1)))

        total_loss = np.mean(total_loss)
        fscore = np.mean(fscore)

        visualize_embedding(z_train, z_aug, z_test, y_test)

    return total_loss, fscore


def visualize_embedding(z_train, z_aug, z_test, y_test):
    y_test_t = np.max(y_test, axis=1)
    xt = TSNE(n_components=2, random_state=42).fit_transform(torch.cat([z_train, z_aug, z_test], dim=0).cpu().numpy())
    
    plt.figure(figsize=(8, 6))
    plt.scatter(xt[:len(z_train), 0], xt[:len(z_train), 1], c='b', alpha=0.5, label='Train Normal')
    plt.scatter(xt[len(z_train):len(z_train)+len(z_aug), 0],
                xt[len(z_train):len(z_train)+len(z_aug), 1], c='orange', alpha=0.5, label='Train Augmented')
    plt.scatter(xt[np.where(y_test_t == 0)[0] + len(z_train) + len(z_aug), 0],
                xt[np.where(y_test_t == 0)[0] + len(z_train) + len(z_aug), 1], c='g', alpha=0.5, label='Test Normal')
    plt.scatter(xt[np.where(y_test_t == 1)[0] + len(z_train) + len(z_aug), 0],
                xt[np.where(y_test_t == 1)[0] + len(z_train) + len(z_aug), 1], c='r', alpha=0.5, label='Test Anomaly (Platform)')
    
    plt.legend()
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig('embedding_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.show()
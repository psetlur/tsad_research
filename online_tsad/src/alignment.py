import logging
import math
from geomloss import SamplesLoss
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn, optim

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


def inject_platform_sgd(ts_row, level, start, length):
    len_ts = len(ts_row)
    start_idx = start * len_ts
    end_idx = start_idx + length * len_ts
    indices = torch.arange(len_ts, device=ts_row.device)
    temperature = 0.1
    mask_left = torch.sigmoid((indices - start_idx) / temperature)
    mask_right = torch.sigmoid((end_idx - indices) / temperature)
    mask = mask_left * mask_right
    x_aug = ts_row * (1 - mask) + level * mask
    label = mask.detach().cpu().numpy()
    return x_aug, label


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
    ts_row = ((ts_row - ts_row.min()) / (ts_row.max() - ts_row.min())) * 2 - 1
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


def black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point, train_point=None):
    ratio_anomaly = 0.1
    valid_level = valid_point['level']
    valid_length = torch.tensor(valid_point['length'], device=args.device)
    anomaly_type = 'platform'

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
        x_valid_aug, valid_labels = list(), list()
        for i in valid_outlier_index:
            x_aug, l = inject(anomaly_type=anomaly_type, ts=x_valid_np[i],
                              config=[valid_level, np.random.uniform(0, 0.5), valid_length])
            x_valid_aug.append(x_aug)
            valid_labels.append(l)

        emb = EmbNormalizer()
        z_valid_aug = model(torch.tensor(np.array(x_valid_aug)).float().unsqueeze(1).to(args.device)).detach()
        z_train_t, z_valid_t, z_valid_aug_t = emb(z_train[train_inlier_index].clone().squeeze(),
                                                  z_valid[valid_inlier_index].clone().squeeze(), z_valid_aug)

    if train_point == None:
        train_level = torch.tensor(0.0, requires_grad=True, device=args.device)
        train_length = torch.tensor(0.0, requires_grad=True, device=args.device)
        optimizer = optim.Adam([train_level, train_length], lr=0.1)

        total_loss, f1score, points = list(), list(), list()
        best = {'level': -1.0, 'length': 0.2, 'wd': np.inf, 'f1-score': 0}
        for _ in range(100):
            train_level = torch.sigmoid(train_level)
            train_length = 0.2 + 0.3 * torch.sigmoid(train_length)
            x_train_aug, train_labels = list(), list()
            for i in train_outlier_index:
                x_aug, l = inject_platform_sgd(torch.tensor(x_train_np[i]).to(args.device), train_level,
                                               np.random.uniform(0, 0.5), train_length)
                x_train_aug.append(x_aug)
                train_labels.append(l)
            z_train_aug = model(torch.stack(x_train_aug, dim=0).float().unsqueeze(1).to(args.device))
            z_train_aug_t = emb.normalize(emb=z_train_aug)

            W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
            loss = W_loss(torch.cat([z_train_t, z_train_aug_t], dim=0), torch.cat([z_valid_t, z_valid_aug_t], dim=0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            X = torch.cat([z_train_t, z_train_aug_t.detach()], dim=0)
            y = torch.tensor(np.concatenate([np.zeros((len(train_inlier_index), x_train_np.shape[1])),
                                             train_labels], axis=0)).to(args.device)
            classify_model = train_classify_model(args=args, X_train=X, y_train=y)
            y_pred = classify(model=classify_model, X_valid=z_valid_aug_t.detach())
            f1 = f1_score(np.array(valid_labels, dtype=np.int64).reshape(-1), y_pred.reshape(-1))
            print(f'train.level.length: {train_level.item()}.{train_length.item()}, '
                  f'valid.level.length: {valid_level}.{valid_length}, wd: {loss.item()}, f1-score: {f1}')
            total_loss.append(loss.item())
            f1score.append(f1)
            points.append({'level': torch.sigmoid(train_level).item(),
                           'length': 0.2 + 0.3 * torch.sigmoid(train_length).item()})
            if loss < best['wd']:
                best = {'level': train_level.item(), 'length': train_length.item(), 'wd': loss, 'f1-score': f1}
            return total_loss, f1score, points, best
    else:
        train_level = train_point['level']
        train_length = train_point['length']

        x_train_aug, train_labels = list(), list()
        for i in train_outlier_index:
            x_aug, l = inject(anomaly_type=anomaly_type, ts=x_train_np[i],
                              config=[train_level, np.random.uniform(0, 0.5), train_length])
            x_train_aug.append(x_aug)
            train_labels.append(l)
        z_train_aug = model(torch.tensor(np.array(x_train_aug)).float().unsqueeze(1).to(args.device))
        z_train_aug_t = emb.normalize(emb=z_train_aug)

        W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
        loss = W_loss(torch.cat([z_train_t, z_train_aug_t], dim=0), torch.cat([z_valid_t, z_valid_aug_t], dim=0))

        X = torch.cat([z_train_t, z_train_aug_t.detach()], dim=0)
        y = torch.tensor(np.concatenate([np.zeros((len(train_inlier_index), x_train_np.shape[1])),
                                         train_labels], axis=0)).to(args.device)
        classify_model = train_classify_model(args=args, X_train=X, y_train=y)
        y_pred = classify(model=classify_model, X_valid=z_valid_aug_t.detach())
        f1score = f1_score(torch.tensor(np.array(valid_labels)).reshape(-1), y_pred.reshape(-1))
        return loss.item(), f1score

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
            emb_x = emb_x.view(emb_x.size(0), -1)
            emb_y = emb_y.view(emb_y.size(0), -1)
            emb_z = emb_z.view(emb_z.size(0), -1)

            emb_all = torch.cat([emb_x, emb_y, emb_z], dim=0)
            self.emb_mean = emb_all.mean(0, keepdim=True) 
            if emb_all.size(0) > 1:
                 self.emb_std = torch.norm(emb_all - self.emb_mean, dim=1).mean() / math.sqrt(emb_all.size(1)) + 1e-6 # Normalize by feature dim instead? Let's try original way adjusted
                 self.emb_std = torch.norm(emb_all - self.emb_mean) / math.sqrt(emb_all.size(0) * emb_all.size(1)) + 1e-6 # Use total elements
            else:
                 self.emb_std = torch.ones_like(self.emb_mean) 

            emb_x = (emb_x - self.emb_mean) / self.emb_std
            emb_y = (emb_y - self.emb_mean) / self.emb_std
            emb_z = (emb_z - self.emb_mean) / self.emb_std
            return emb_x, emb_y, emb_z
        else:
            raise ValueError(self.mode)

    def normalize(self, emb):
        if self.emb_mean is None or self.emb_std is None:
            raise ValueError("Normalizer has not been fitted. Call it first.")
        emb = emb.view(emb.size(0), -1) 
        return (emb - self.emb_mean) / self.emb_std



def inject_platform(ts_row, level, start, length):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    start_a = int(len(ts_row) * start)
    length_a = int(len(ts_row) * length)
    end_a = min(start_a + length_a, len(ts_row)) # Ensure end index is valid
    ts_row[start_a: end_a] = level
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

def inject_spike(ts_row, level, p):
    ts_row = np.array(ts_row)
    label = np.zeros(len(ts_row))
    mask = bernoulli.rvs(p=p, size=len(ts_row)).astype(bool)
    if np.any(mask):
        modified_values = ts_row[mask] * level
        modified_values[(ts_row[mask] > 0) & (modified_values < 1)] = 1
        modified_values[(ts_row[mask] < 0) & (modified_values > -1)] = -1
        ts_row[mask] = modified_values
        label[mask] = 1
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
    ts_row[start_a : end_a] += slope_a
    if slope_a.size > 0 and end_a < ts_len:
        ts_row[end_a:] += np.full(ts_len - end_a, slope_a[-1])
    label[start_a : end_a] = 1
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



def train_classify_model(args, X_train, y_train, sequence_length):

    embed_dim = X_train.shape[1]
    model = nn.Sequential(
        nn.Linear(embed_dim, 128),
        nn.ReLU(),
        nn.Linear(128, sequence_length) # Output matches sequence length
    ).to(args.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

    num_epochs = 1000 
    batch_size = 64
    dataset = torch.utils.data.TensorDataset(X_train, y_train.float()) # Ensure y is float
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    model.eval()
    return model

def classify(model, X_valid):

    model.eval()
    with torch.no_grad():
        logits = model(X_valid)
        probs = torch.sigmoid(logits)
        y_pred = torch.where(probs > 0.5, 1, 0).cpu().numpy()
    return y_pred


def black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point,
                       valid_anomaly_types, train_point, best=False, calculate_f1 = True):
    ratio_anomaly = 0.1
    model.eval()
    anomaly_types = ['platform', 'mean', 'spike', 'amplitude', 'trend', 'variance']

    with torch.no_grad():
        z_train_list, x_train_list = [], []
        for x_batch in train_dataloader:
            c_x = model(x_batch.to(args.device)).detach()
            z_train_list.append(c_x)
            x_train_list.append(x_batch.cpu().numpy()) # Move to CPU before numpy
        z_train = torch.cat(z_train_list, dim=0)
        x_train_np = np.concatenate(x_train_list, axis=0)
        if x_train_np.ndim == 3 and x_train_np.shape[1] == 1:
             x_train_np = x_train_np.squeeze(1)
        elif x_train_np.ndim != 2:
             raise ValueError(f"Unexpected shape for x_train_np: {x_train_np.shape}")
        sequence_length = x_train_np.shape[1] # Get sequence length

        z_valid_list, x_valid_list = [], []
        for x_batch in val_dataloader:
            c_x = model(x_batch.to(args.device)).detach()
            z_valid_list.append(c_x)
            x_valid_list.append(x_batch.cpu().numpy())
        z_valid = torch.cat(z_valid_list, dim=0)
        x_valid_np = np.concatenate(x_valid_list, axis=0)
        if x_valid_np.ndim == 3 and x_valid_np.shape[1] == 1:
             x_valid_np = x_valid_np.squeeze(1) # Reshape to [N, SeqLen]
        elif x_valid_np.ndim != 2:
             raise ValueError(f"Unexpected shape for x_valid_np: {x_valid_np.shape}")

    train_inlier_index, train_outlier_index = train_test_split(range(len(x_train_np)),
                                                               train_size=1 - ratio_anomaly, random_state=0)
    valid_inlier_index, valid_outlier_index = train_test_split(range(len(x_valid_np)),
                                                               train_size=1 - ratio_anomaly, random_state=0)

    x_valid_aug, valid_labels = list(), list()
    for i in valid_outlier_index:
        temp_x = x_valid_np[i].copy()
        temp_label = np.zeros(sequence_length)
        for anomaly_type in valid_anomaly_types:
            if anomaly_type == 'platform':
                temp_x, l = inject_platform(temp_x, valid_point[anomaly_type]['level'], np.random.uniform(0, 1 - valid_point[anomaly_type]['length']), valid_point[anomaly_type]['length'])
            elif anomaly_type == 'mean':
                temp_x, l = inject_mean(temp_x, valid_point[anomaly_type]['level'], np.random.uniform(0, 1 - valid_point[anomaly_type]['length']), valid_point[anomaly_type]['length'])
            elif anomaly_type == 'spike':
                temp_x, l = inject_spike(temp_x, valid_point[anomaly_type]['level'], valid_point[anomaly_type]['p'])
            elif anomaly_type == 'amplitude':
                temp_x, l = inject_amplitude(temp_x, valid_point[anomaly_type]['level'], np.random.uniform(0, 1 - valid_point[anomaly_type]['length']), valid_point[anomaly_type]['length'])
            elif anomaly_type == 'trend':
                temp_x, l = inject_trend(temp_x, valid_point[anomaly_type]['slope'], np.random.uniform(0, 1 - valid_point[anomaly_type]['length']), valid_point[anomaly_type]['length'])
            elif anomaly_type == 'variance':
                temp_x, l = inject_variance(temp_x, valid_point[anomaly_type]['level'], np.random.uniform(0, 1 - valid_point[anomaly_type]['length']), valid_point[anomaly_type]['length'])
            else:
                 raise ValueError(f'Unsupported anomaly_type in valid_point: {anomaly_type}')
            temp_label = np.logical_or(temp_label, l).astype(int) # Combine labels if multiple anomalies injected
        x_valid_aug.append(temp_x)
        valid_labels.append(temp_label)

    # Parse BO suggested point
    train_p = {}
    for k, v in train_point.items():
        s = k.split('_')
        anomaly_type = s[0]
        param_name = s[1]
        if anomaly_type not in train_p:
            train_p[anomaly_type] = {}
        train_p[anomaly_type][param_name] = v

    x_train_aug, train_labels = list(), list()
    
    for i in train_outlier_index:
        temp_x = x_train_np[i].copy() 
        temp_label = np.zeros(sequence_length)
        for anomaly_type in anomaly_types:
            if anomaly_type == 'platform':
                 if train_p[anomaly_type]['level'] != 0 or train_p[anomaly_type]['length'] != 0:
                    temp_x, l = inject_platform(temp_x, train_p[anomaly_type]['level'], np.random.uniform(0, 1 - train_p[anomaly_type]['length']), train_p[anomaly_type]['length'])
                    temp_label = np.logical_or(temp_label, l).astype(int)
            elif anomaly_type == 'mean':
                 if train_p[anomaly_type]['level'] != 0 or train_p[anomaly_type]['length'] != 0:
                    temp_x, l = inject_mean(temp_x, train_p[anomaly_type]['level'], np.random.uniform(0, 1 - train_p[anomaly_type]['length']), train_p[anomaly_type]['length'])
                    temp_label = np.logical_or(temp_label, l).astype(int)
            elif anomaly_type == 'spike':
                 if train_p[anomaly_type]['level'] != 0 or train_p[anomaly_type]['p'] != 0:
                    temp_x, l = inject_spike(temp_x, train_p[anomaly_type]['level'], train_p[anomaly_type]['p'])
                    temp_label = np.logical_or(temp_label, l).astype(int)
            elif anomaly_type == 'amplitude':
                 if train_p[anomaly_type]['level'] != 0 or train_p[anomaly_type]['length'] != 0:
                    temp_x, l = inject_amplitude(temp_x, train_p[anomaly_type]['level'], np.random.uniform(0, 1 - train_p[anomaly_type]['length']), train_p[anomaly_type]['length'])
                    temp_label = np.logical_or(temp_label, l).astype(int)
            elif anomaly_type == 'trend':
                 if train_p[anomaly_type]['slope'] != 0 or train_p[anomaly_type]['length'] != 0:
                    temp_x, l = inject_trend(temp_x, train_p[anomaly_type]['slope'], np.random.uniform(0, 1 - train_p[anomaly_type]['length']), train_p[anomaly_type]['length'])
                    temp_label = np.logical_or(temp_label, l).astype(int)
            elif anomaly_type == 'variance':
                 if train_p[anomaly_type]['level'] != 0 or train_p[anomaly_type]['length'] != 0:
                    temp_x, l = inject_variance(temp_x, train_p[anomaly_type]['level'], np.random.uniform(0, 1 - train_p[anomaly_type]['length']), train_p[anomaly_type]['length'])
                    temp_label = np.logical_or(temp_label, l).astype(int)
            else:
                 raise ValueError(f'Unsupported anomaly_type during train augmentation: {anomaly_type}')
        x_train_aug.append(temp_x)
        train_labels.append(temp_label)

    x_train_aug_tensor = torch.tensor(np.array(x_train_aug), dtype=torch.float32).unsqueeze(1).to(args.device)
    x_valid_aug_tensor = torch.tensor(np.array(x_valid_aug), dtype=torch.float32).unsqueeze(1).to(args.device)

    with torch.no_grad():
        z_train_aug = model(x_train_aug_tensor).detach()
        z_valid_aug = model(x_valid_aug_tensor).detach()

    emb = EmbNormalizer()
    z_train_inliers = z_train[train_inlier_index].clone() # Use clone to avoid modifying original
    z_valid_inliers = z_valid[valid_inlier_index].clone()

    z_train_inliers = z_train_inliers.view(z_train_inliers.size(0), -1)
    z_valid_inliers = z_valid_inliers.view(z_valid_inliers.size(0), -1)
    z_train_aug_flat = z_train_aug.view(z_train_aug.size(0), -1)
    z_valid_aug_flat = z_valid_aug.view(z_valid_aug.size(0), -1)

    z_train_t, z_valid_t, _ = emb(z_train_inliers,
                                  z_valid_inliers,
                                  torch.cat([z_train_aug_flat, z_valid_aug_flat], dim=0)) 

    z_train_aug_t = emb.normalize(emb=z_train_aug_flat)
    z_valid_aug_t = emb.normalize(emb=z_valid_aug_flat)


    if best is False:
        W_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)

        loss_tensor = W_loss(z_train_aug_t, z_valid_aug_t)
        loss = loss_tensor.item()

        f1score_val = 0.0 # Default F1 score

        if calculate_f1:
            X_clf_train = torch.cat([z_train_t, z_train_aug_t], dim=0).detach()

            train_labels_np = np.array(train_labels)
            valid_labels_np = np.array(valid_labels)

            if train_labels_np.size > 0 and train_labels_np.shape[1] == sequence_length and \
               valid_labels_np.size > 0 and valid_labels_np.shape[1] == sequence_length:

                y_clf_train_np = np.concatenate([np.zeros((len(train_inlier_index), sequence_length)),
                                             train_labels_np], axis=0)
                y_clf_train = torch.tensor(y_clf_train_np, dtype=torch.float32).to(args.device)

                classify_model = train_classify_model(args=args, X_train=X_clf_train, y_train=y_clf_train, sequence_length=sequence_length)

                y_pred = classify(model=classify_model, X_valid=z_valid_aug_t.detach())

                if y_pred.shape == valid_labels_np.shape:
                     f1score_val = f1_score(valid_labels_np.reshape(-1), y_pred.reshape(-1), zero_division=0)
                else:
                     logging.warning(f"F1 Calc: Shape mismatch between valid labels {valid_labels_np.shape} and predictions {y_pred.shape}. F1=0.")
            else:
                logging.warning(f"F1 Calc: Problem with label shapes/sizes. Train: {train_labels_np.shape}, Valid: {valid_labels_np.shape}, Expected SeqLen: {sequence_length}. F1=0.")

        return loss, f1score_val

    else: 
        log_dir = f'logs/training/{args.trail}'
        os.makedirs(log_dir, exist_ok=True)
        visualize(z_train_t.cpu(), z_valid_t.cpu(), z_train_aug_t.cpu(), z_valid_aug_t.cpu())
        return None, None # Match return signature



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

    if all_data.shape[0] <= 1: # TSNE needs more than 1 sample
         logging.warning(f"Visualize: Not enough samples ({all_data.shape[0]}) for t-SNE.")
         return

    xt = TSNE(n_components=2, random_state=42, perplexity=min(30, all_data.shape[0] - 1)).fit_transform(all_data) # Adjust perplexity

    plt.figure(figsize=(8, 6))
    start_idx = 0
    if train_np.size > 0:
        plt.scatter(xt[start_idx:start_idx + len(train_np), 0], xt[start_idx:start_idx + len(train_np), 1], c='b', alpha=0.5, label='Train Inlier')
        start_idx += len(train_np)
    if test_np.size > 0:
        plt.scatter(xt[start_idx:start_idx + len(test_np), 0], xt[start_idx:start_idx + len(test_np), 1], c='g', alpha=0.5, label='Valid Inlier')
        start_idx += len(test_np)
    if train_aug_np.size > 0:
        plt.scatter(xt[start_idx:start_idx + len(train_aug_np), 0], xt[start_idx:start_idx + len(train_aug_np), 1], c='orange', alpha=0.5, label='Train Aug (BO Point)')
        start_idx += len(train_aug_np)
    if test_outlier_np.size > 0:
        plt.scatter(xt[start_idx:start_idx + len(test_outlier_np), 0], xt[start_idx:start_idx + len(test_outlier_np), 1], c='r', alpha=0.5, label='Valid Aug (Target Point)')

    plt.legend()
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.title('t-SNE of Embeddings (Final Best Point vs Target)')
    log_dir = 'logs/viz' # Example placeholder
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(f'{log_dir}/tsne_visualization_best.pdf', dpi=300, bbox_inches='tight')
    # plt.show() # Avoid showing plots in non-interactive environments
    plt.close()


def visualize_time_series(time_series):
    plt.figure(figsize=(10, 4)) # Smaller figure
    plt.plot(time_series, label="Time Series", color="blue", linewidth=1)
    plt.title("Time Series Visualization", fontsize=14)
    plt.xlabel("Time Step", fontsize=10)
    plt.ylabel("Value", fontsize=10)
    plt.grid(alpha=0.3)
    plt.close() 
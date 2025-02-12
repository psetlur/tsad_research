import numpy as np
import torch
import pytorch_lightning as pl
from info_nce import InfoNCE
from .model import CNNEncoder
import random

LENGTH_BINS = [0.2, 0.3, 0.4, 0.5]
LEVEL_BINS = [-1, -0.33, 0.33, 1]
TAU = 0.01
TAU_LEVEL = 0.01
TAU_LENGTH = 0.002
NEG_RANGE = -0.5
POS_RANGE = 0.5

FIXED_LEVEL = 0.5
FIXED_LENGTH = 0.3

GRID_LEVEL = np.round(np.arange(-1, 1.1, 0.1), 1)
GRID_LENGTH = np.round(np.arange(0.2, 0.52, 0.02), 2)
CDF_LEVEL = np.arange(0, 1, 1 / len(GRID_LEVEL))
CDF_LENGTH = np.arange(0, 1, 1 / len(GRID_LENGTH))

step_size = 3
sampled_levels = GRID_LEVEL[::step_size]
sampled_lengths = GRID_LENGTH[::step_size]

negatives = [(l, random.choice(np.arange(0, 0.5, 0.01)) + np.random.uniform(-TAU, TAU), le)
             for l in sampled_levels for le in sampled_lengths]
print(f"Total negatives selected: {len(negatives)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
math explanation - 21 points in GRID_LEVEL, 16 points in GRID_LENGTH, so there are 21 * 16 = 336 grid cells
336 is too large and will make training too slow, so we randomly sample the grid into a few broad regions and pick negatives
from each region to maintain diversity
'''


def hist_sample(cdf, bins):
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    val = np.random.uniform(bins[bin_idx - 1], bins[bin_idx])
    return val

# def hard_negative_loss(z_anc, z_pos, z_neg, meta_anc, meta_neg, temperature=0.1):
#     meta_anc = torch.tensor(meta_anc).to(z_anc.device)
#     meta_neg = torch.tensor(meta_neg).to(z_anc.device)
#     neg_weights = torch.cdist(meta_anc, meta_neg, p=1)
#
#     # neg score weighted by absolute distance of hyperparameters
#     neg = torch.exp(neg_weights * torch.mm(z_anc, z_neg.t().contiguous()) / temperature).sum(dim=-1)
#
#     # pos score
#     pos = torch.exp(torch.sum(z_anc * z_pos, dim=-1) / temperature)
#
#     # contrastive loss
#     loss = (-torch.log(pos / (pos + neg))).mean()
#     return loss

def hard_negative_loss(z_anc, z_pos, z_neg, meta_anc, meta_neg, temperature=0.1):
    meta_anc = torch.tensor(meta_anc).to(device)
    meta_neg = torch.tensor(meta_neg).to(device)

    if meta_anc.dim() == 1:
        meta_anc = meta_anc.unsqueeze(0)
    if meta_neg.dim() == 1:
        meta_neg = meta_neg.unsqueeze(0)

    z_anc_flat = z_anc.view(-1, z_anc.size(-1)) if z_anc.dim() == 3 else z_anc
    z_neg_flat = z_neg.view(-1, z_neg.size(-1)) if z_neg.dim() == 3 else z_neg

    neg_weights = torch.cdist(meta_anc, meta_neg, p=1)

    # neg score weighted by absolute distance of hyperparameters
    neg = torch.exp(neg_weights * torch.mm(z_anc_flat, z_neg_flat.t().contiguous()) / temperature).sum(dim=-1)

    # pos score
    pos = torch.exp(torch.sum(z_anc_flat * z_pos, dim=-1) / temperature)

    # contrastive loss
    loss = (-torch.log(pos / (pos + neg))).mean()
    return loss


def config_from_grid(cdf, grid):
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    return grid[bin_idx - 1]


class Encoder(pl.LightningModule):
    def __init__(self, args, ts_input_size, lr, a_config):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.a_config = a_config

        self.encoder = CNNEncoder(ts_input_size)
        self.lr = lr
        self.info_loss = InfoNCE(negative_mode='unpaired')

        self.normal_idx = set()
        self.normal_x = torch.tensor([]).to(device)

    def forward(self, x):
        x = self.encoder(x)
        return x

    def inject_platform(self, ts_row, level, start, length):
        start_a = int(len(ts_row) * start)
        length_a = int(len(ts_row) * length)
        ts_row[start_a: start_a + length_a] = float(level)
        return ts_row

    def inject_mean(self, ts_row, level, start, length):
        start = int(len(ts_row) * start)
        length = int(len(ts_row) * length)
        ts_row[start: start + length] += float(level)
        return ts_row

    def training_step(self, x, batch_idx):
        x = x.to(device)
        self.normal_x.to(device)
        if batch_idx not in self.normal_idx:
            self.normal_idx.add(batch_idx)
            self.normal_x = torch.cat([self.normal_x, x], dim=0).to(device)

        # multiple positive samples
        num_positives = 3
        y_pos = [x.clone().to(device) for _ in range(num_positives)]
        meta_pos = []

        # multiple negative samples
        num_negatives = len(negatives)
        y_neg = [x.clone().to(device) for _ in range(num_negatives)]
        meta_neg = []

        for i in range(len(x)):
            # First platform anomaly - original
            m0 = [config_from_grid(CDF_LEVEL, GRID_LEVEL),
                  random.choice(np.arange(0, 0.5, 0.01)),
                  config_from_grid(CDF_LENGTH, GRID_LENGTH)]

            # generate positive samples
            for j in range(num_positives):
                pos_variation = [
                    m0[0] + np.random.uniform(-TAU, TAU),
                    max(m0[1] + np.random.uniform(-TAU, TAU), 0),
                    max(m0[2] + np.random.uniform(-TAU, TAU), 0)
                ]
                y_pos[j][i][0] = self.inject_platform(y_pos[j][i][0], *pos_variation)
                meta_pos.append(pos_variation)

            # generating diverse negative samples
            sampled_negatives = random.sample(negatives, num_negatives)

            # generate negative samples
            for j in range(num_negatives):
                neg_variation = sampled_negatives[j]  # unique grid location
                y_neg[j][i][0] = self.inject_platform(y_neg[j][i][0], *neg_variation)
                meta_neg.append(neg_variation)

        # concatenating samples
        all_samples = torch.cat([x] + y_pos + y_neg, dim=0).to(device)
        outputs = self(all_samples)

        # Validate sizes for splitting
        total_expected_size = x.shape[0] * (1 + num_positives + num_negatives)
        if outputs.shape[0] != total_expected_size:
            raise ValueError(f"Mismatch in tensor size. Expected {total_expected_size}, but got {outputs.shape[0]}.")

        split_sizes = [x.shape[0]] + [x.shape[0]] * num_positives + [x.shape[0]] * num_negatives
        split_outputs = torch.split(outputs, split_sizes, dim=0)

        c_x = split_outputs[0]
        c_y_pos = split_outputs[1:num_positives + 1]
        c_y_neg = split_outputs[num_positives + 1:]

        if self.current_epoch < 30:
            weight_normal = 1.0
            weight_global = 0.01
            weight_local = 0.001
        elif self.current_epoch < 60:
            weight_normal = 1.0
            weight_global = 1.0
            weight_local = 0.01
        else:
            weight_normal = 1.0
            weight_global = 1.0
            weight_local = 1.0

        loss_global = sum(self.info_loss(c_x, c_y_p, torch.cat([c_x] + [c_x], dim=0)) for c_y_p in c_y_pos) / len(c_y_pos)
        
        ### Anomalies with far away hyperparameters should be far away propotional to delta.
        loss_local = sum(hard_negative_loss(c_x, c_y_p, torch.stack(c_y_neg), meta_pos[i], meta_neg[i]) 
                         for i, c_y_p in enumerate(c_y_pos)) / len(c_y_pos)

        ### Nomral should be close to each other, and far away from anomalies.
        loss_normal = self.info_loss(c_x, torch.cat(c_y_pos, dim=0), torch.cat([c_x] + list(c_y_pos), dim=0))

        loss = weight_global * loss_global + weight_local * loss_local + weight_normal * loss_normal
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, x, batch_idx):
        x = x.to(device)
        x_pos = self.normal_x[np.random.choice(len(self.normal_x), x.shape[0])].to(device)

        anomalies_start = random.choices([i for i in np.arange(0, 0.5, 0.01)], k=len(x))

        num_positives = 3
        num_negatives = len(negatives)

        y_pos = [x.clone().to(device) for _ in range(num_positives)]  # Multiple positive samples
        y_neg = [x.clone().to(device) for _ in range(num_negatives)]  # Multiple negative samples
        meta_pos = []
        meta_neg = []

        for i in range(len(x)):
            ### Platform anomaly
            m0 = [config_from_grid(CDF_LEVEL, GRID_LEVEL), anomalies_start[i],
                  config_from_grid(CDF_LENGTH, GRID_LENGTH)]
            y_0 = x.clone()
            y_0[i][0] = self.inject_platform(y_0[i][0], *m0)

            # positive sample
            for j in range(num_positives):
                pos_variation = [
                    m0[0] + np.random.uniform(-TAU, TAU),
                    max(m0[1] + np.random.uniform(-TAU, TAU), 0),
                    max(m0[2] + np.random.uniform(-TAU, TAU), 0)
                ]
                y_pos[j][i][0] = self.inject_platform(y_pos[j][i][0], *pos_variation)
                meta_pos.append(pos_variation)

            # generating diverse negative samples
            sampled_negatives = random.sample(negatives, num_negatives)

            # generate negative samples
            for j in range(num_negatives):
                neg_variation = sampled_negatives[j]  # unique grid location
                y_neg[j][i][0] = self.inject_platform(y_neg[j][i][0], *neg_variation)
                meta_neg.append(neg_variation)

        all_samples = torch.cat([x] + y_pos + y_neg + [x_pos], dim=0).to(device)
        outputs = self(all_samples)

        # Validate output sizes and split outputs
        total_expected_size = x.shape[0] * (1 + num_positives + num_negatives + 1)
        if outputs.shape[0] != total_expected_size:
            raise ValueError(f"Mismatch in tensor size. Expected {total_expected_size}, but got {outputs.shape[0]}.")

        split_sizes = [x.shape[0]] + [x.shape[0]] * num_positives + [x.shape[0]] * num_negatives + [x.shape[0]]
        split_outputs = torch.split(outputs, split_sizes, dim=0)

        c_x = split_outputs[0]
        c_y_pos = split_outputs[1:num_positives + 1]
        c_y_neg = split_outputs[num_positives + 1:num_positives + 1 + num_negatives]
        c_x_pos = split_outputs[-1]

        loss_global = sum(self.info_loss(c_x, c_y_p, torch.cat([c_x] + [c_x], dim=0)) for c_y_p in c_y_pos) / len(c_y_pos)

        loss_local = sum(hard_negative_loss(c_x, c_y_p, torch.stack(c_y_neg), meta_pos[i], meta_neg[i]) 
                         for i, c_y_p in enumerate(c_y_pos)) / len(c_y_pos)

        loss_normal = self.info_loss(c_x, torch.cat(c_y_pos, dim=0), torch.cat([c_x] + list(c_y_pos), dim=0))

        
        loss = loss_global + loss_local + loss_normal
        self.log("loss_global", loss_global, prog_bar=True)
        self.log("loss_local", loss_local, prog_bar=True)
        self.log("loss_normal", loss_normal, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    # def training_step(self, x, batch_idx):
    #     if batch_idx not in self.normal_idx:
    #         self.normal_idx.add(batch_idx)
    #         self.normal_x = torch.cat([self.normal_x, x], dim=0)
    #     x_pos = self.normal_x[np.random.choice(len(self.normal_x), x.shape[0])]
    #     if self.current_epoch == 110:
    #         print(1)
    #     y, y_pos, y_neg = x.clone(), x.clone(), x.clone()
    #     meta, meta_pos, meta_neg = list(), list(), list()
    #
    #     for i in range(len(x)):
    #         ### Platform anomaly
    #         # m = [hist_sample(level_0_cdf, LEVEL_BINS), np.random.uniform(0, 0.5), hist_sample(length_0_cdf, LENGTH_BINS)]
    #         # m = [FIXED_LEVEL, np.random.uniform(0, 0.5), FIXED_LENGTH]
    #         if self.args.trail == 'fixed':
    #             m = [FIXED_LEVEL, np.random.uniform(0, 0.5), FIXED_LENGTH]
    #         elif self.args.trail == 'grid' or self.args.trail == 'more_epochs' or self.args.trail == 'loss_length_tau':
    #             m = [config_from_grid(CDF_LEVEL, GRID_LEVEL), np.random.uniform(0, 0.5),
    #                  config_from_grid(CDF_LENGTH, GRID_LENGTH)]
    #         else:
    #             raise Exception('Unsupported trail.')
    #         y[i][0] = self.inject_platform(y[i][0], *m)
    #         meta.append(m)
    #
    #         # positive samples
    #         s1 = np.random.uniform(0, 0.5)
    #         if self.args.trail == 'loss_length_tau':
    #             s0 = max(m[0] + np.random.uniform(low=-TAU_LEVEL, high=TAU_LEVEL), -1.0)
    #             s0 = min(s0, 1.0)
    #             s2 = max(m[2] + np.random.uniform(low=-TAU_LENGTH, high=TAU_LENGTH), 0.20)
    #             s2 = min(s2, 0.50)
    #         else:
    #             s0 = m[0] + np.random.uniform(low=-TAU, high=TAU)
    #             s2 = max(m[2] + np.random.uniform(low=-TAU, high=TAU), 0)
    #
    #         y_pos[i][0] = self.inject_platform(y_pos[i][0], s0, s1, s2)
    #         meta_pos.append([s0, s1, s2])
    #
    #         # negative samples
    #         s1_neg = np.random.uniform(0, 0.5)
    #         if self.args.trail == 'loss_length_tau':
    #             s0_neg = np.random.uniform(low=-1.0, high=m[0] - TAU_LEVEL) if np.random.random() < (
    #                     (m[0] + 1.0) / 2.0) else np.random.uniform(low=m[0] + TAU_LEVEL, high=1.0)
    #             s2_neg = np.random.uniform(low=0.20, high=m[2] - TAU_LENGTH) if np.random.random() < (
    #                     (m[2] - 0.20) / 0.30) else np.random.uniform(low=m[2] + TAU_LENGTH, high=0.50)
    #         else:
    #             s0_neg = m[0] + np.random.uniform(low=-RANGE, high=-TAU) \
    #                 if np.random.random() > 0.5 else m[0] + np.random.uniform(low=TAU, high=RANGE)
    #             s2_neg = max(m[2] + np.random.uniform(low=-RANGE, high=-TAU) \
    #                              if np.random.random() > 0.5 else m[2] + np.random.uniform(low=TAU, high=RANGE), 0)
    #
    #         y_neg[i][0] = self.inject_platform(y_neg[i][0], s0_neg, s1_neg, s2_neg)
    #         meta_neg.append([s0_neg, s1_neg, s2_neg])
    #
    #     outputs = self(torch.cat([x, y, y_pos, y_neg, x_pos], dim=0))
    #     c_x, c_y, c_y_pos, c_y_neg, c_x_pos = torch.split(outputs, x.shape[0], dim=0)
    #
    #     ### Anomalies should be close to the ones with the same type and similar hyperparameters, and far away from the ones with different types and normal.
    #     loss_global = self.info_loss(c_y, c_y_pos, torch.cat([c_x, c_x_pos], dim=0))
    #
    #     ### Anomalies with far away hyperparameters should be far away propotional to delta.
    #     if self.args.trail == 'loss_length_tau':
    #         loss_local = self.info_loss(c_y, c_y_pos, c_y_neg)
    #     else:
    #         loss_local = hard_negative_loss(c_y, c_y_pos, c_y_neg, np.array(meta), np.array(meta_neg))
    #
    #     ### Nomral should be close to each other, and far away from anomalies.
    #     loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([c_y, c_y_pos, c_y_neg], dim=0))
    #
    #     loss = loss_global + loss_local + loss_normal
    #
    #     if loss_global > 0:
    #         pass
    #     else:
    #         raise Exception('!!!')
    #     self.log("loss_global", loss_global, prog_bar=True)
    #     self.log("loss_local", loss_local, prog_bar=True)
    #     self.log("loss_normal", loss_normal, prog_bar=True)
    #     self.log("train_loss", loss, prog_bar=True)
    #     return loss
    #
    # def validation_step(self, x, batch_idx):
    #     x_pos = self.normal_x[np.random.choice(len(self.normal_x), x.shape[0])]
    #
    #     y, y_pos, y_neg = x.clone(), x.clone(), x.clone()
    #     meta, meta_pos, meta_neg = list(), list(), list()
    #
    #     for i in range(len(x)):
    #         ### Platform anomaly
    #         # m = [hist_sample(level_0_cdf, LEVEL_BINS), np.random.uniform(0, 0.5), hist_sample(length_0_cdf, LENGTH_BINS)]
    #         if self.args.trail == 'fixed':
    #             m = [FIXED_LEVEL, np.random.uniform(0, 0.5), FIXED_LENGTH]
    #         elif self.args.trail == 'grid' or self.args.trail == 'more_epochs' or self.args.trail == 'loss_length_tau':
    #             m = [config_from_grid(CDF_LEVEL, GRID_LEVEL), np.random.uniform(0, 0.5),
    #                  config_from_grid(CDF_LENGTH, GRID_LENGTH)]
    #         else:
    #             raise Exception('Unsupported trail.')
    #         y[i][0] = self.inject_platform(y[i][0], *m)
    #         meta.append(m)
    #
    #         # positive sample
    #         s1 = np.random.uniform(0, 0.5)
    #         if self.args.trail == 'loss_length_tau':
    #             s0 = max(m[0] + np.random.uniform(low=-TAU_LEVEL, high=TAU_LEVEL), -1.0)
    #             s0 = min(s0, 1.0)
    #             s2 = max(m[2] + np.random.uniform(low=-TAU_LENGTH, high=TAU_LENGTH), 0.20)
    #             s2 = min(s2, 0.50)
    #         else:
    #             s0 = m[0] + np.random.uniform(low=-TAU, high=TAU)
    #             s2 = max(m[2] + np.random.uniform(low=-TAU, high=TAU), 0)
    #
    #         y_pos[i][0] = self.inject_platform(y_pos[i][0], s0, s1, s2)
    #         meta_pos.append([s0, s1, s2])
    #
    #         # negative samples
    #         s1_neg = np.random.uniform(0, 0.5)
    #         if self.args.trail == 'loss_length_tau':
    #             s0_neg = np.random.uniform(low=-1.0, high=m[0] - TAU_LEVEL) if np.random.random() < (
    #                     (m[0] + 1.0) / 2.0) else np.random.uniform(low=m[0] + TAU_LEVEL, high=1.0)
    #             s2_neg = np.random.uniform(low=0.20, high=m[2] - TAU_LENGTH) if np.random.random() < (
    #                     (m[2] - 0.20) / 0.30) else np.random.uniform(low=m[2] + TAU_LENGTH, high=0.50)
    #         else:
    #             s0_neg = m[0] + np.random.uniform(low=-RANGE, high=-TAU) \
    #                 if np.random.random() > 0.5 else m[0] + np.random.uniform(low=TAU, high=RANGE)
    #             s2_neg = max(m[2] + np.random.uniform(low=-RANGE, high=-TAU) \
    #                              if np.random.random() > 0.5 else m[2] + np.random.uniform(low=TAU, high=RANGE), 0)
    #
    #         y_neg[i][0] = self.inject_platform(y_neg[i][0], s0_neg, s1_neg, s2_neg)
    #         meta_neg.append([s0_neg, s1_neg, s2_neg])
    #
    #     outputs = self(torch.cat([x, y, y_pos, y_neg, x_pos], dim=0))
    #     c_x, c_y, c_y_pos, c_y_neg, c_x_pos = torch.split(outputs, x.shape[0], dim=0)
    #
    #     loss_global = self.info_loss(c_y, c_y_pos, torch.cat([c_x, c_x_pos], dim=0))
    #
    #     if self.args.trail == 'loss_length_tau':
    #         loss_local = self.info_loss(c_y, c_y_pos, c_y_neg)
    #     else:
    #         loss_local = hard_negative_loss(c_y, c_y_pos, c_y_neg, np.array(meta), np.array(meta_neg))
    #
    #     loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([c_y, c_y_pos, c_y_neg], dim=0))
    #
    #     loss = loss_global + loss_local + loss_normal
    #     self.log("loss_global", loss_global, prog_bar=True)
    #     self.log("loss_local", loss_local, prog_bar=True)
    #     self.log("loss_normal", loss_normal, prog_bar=True)
    #     self.log("val_loss", loss, prog_bar=True)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

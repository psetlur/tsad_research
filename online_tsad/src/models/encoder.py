import numpy as np
import torch
import pytorch_lightning as pl
from info_nce import InfoNCE
from .model import CNNEncoder
import random

LENGTH_BINS = [0.2, 0.3, 0.4, 0.5]
LEVEL_BINS = [-1, -0.33, 0.33, 1]
TAU = 0.01
NEG_RANGE = -0.5
POS_RANGE = 0.5

# fixed a config for platform anomaly - trial 1
FIXED_LEVEL = 0.5
FIXED_LENGTH = 0.3
FIXED_START = 0.2

# fixed a grid for platform anomaly - trial 2
GRID_LEVEL = np.round(np.arange(-1, 1.2, 0.2), 1)
GRID_LENGTH = np.round(np.arange(0.2, 0.54, 0.04), 2)
CDF_LEVEL = np.arange(0, 1, 1 / len(GRID_LEVEL))
CDF_LENGTH = np.arange(0, 1, 1 / len(GRID_LENGTH))


def hist_sample(cdf, bins):
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    val = np.random.uniform(bins[bin_idx - 1], bins[bin_idx])
    return val


def hard_negative_loss(z_anc, z_pos, z_neg, meta_anc, meta_neg, temperature=0.1):
    meta_anc = torch.tensor(meta_anc).to(z_anc.device)
    meta_neg = torch.tensor(meta_neg).to(z_anc.device)
    neg_weights = torch.cdist(meta_anc, meta_neg, p=1)

    # neg score weighted by absolute distance of hyperparameters
    neg = torch.exp(neg_weights * torch.mm(z_anc, z_neg.t().contiguous()) / temperature).sum(dim=-1)

    # pos score
    pos = torch.exp(torch.sum(z_anc * z_pos, dim=-1) / temperature)

    # contrastive loss
    loss = (-torch.log(pos / (pos + neg))).mean()
    return loss


def fixed_config_from_grid(cdf, grid):
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    return grid[bin_idx - 1]


class Encoder(pl.LightningModule):
    def __init__(self, ts_input_size, lr, a_config):
        super().__init__()
        self.save_hyperparameters()
        self.a_config = a_config

        self.encoder = CNNEncoder(ts_input_size)
        self.lr = lr
        self.info_loss = InfoNCE(negative_mode='unpaired')

        self.normal_idx = set()
        self.normal_x = torch.tensor([]).to(0)

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
        if batch_idx not in self.normal_idx:
            self.normal_idx.add(batch_idx)
            self.normal_x = torch.cat([self.normal_x, x], dim=0)
        #x_pos = self.normal_x[np.random.choice(len(self.normal_x), x.shape[0])]

        # a_config = self.a_config
        # level_0_cdf  = [0, a_config['level_0_h0'] , a_config['level_0_h0']  + a_config['level_0_h1'] , 1]
        # length_0_cdf = [0, a_config['length_0_h0'], a_config['length_0_h0'] + a_config['length_0_h1'], 1]
        # level_1_cdf  = [0, a_config['level_1_h0'] , a_config['level_1_h0']  + a_config['level_1_h1'] , 1]
        # length_1_cdf = [0, a_config['length_1_h0'], a_config['length_1_h0'] + a_config['length_1_h1'], 1]

        # y_0, y_1 = x.clone(), x.clone()
        # y_0_pos, y_1_pos = x.clone(), x.clone()
        # y_0_neg, y_1_neg = x.clone(), x.clone()
        # meta_0, meta_1, meta_0_neg, meta_1_neg = [], [], [], []

        # # adding another anomaly - trial 2
        # fixed_level_1 = 0.7
        # fixed_length_1 = 0.4
        # fixed_start_1 = 0.7

        #anomalies_start = random.choices([i for i in np.arange(0, 0.5, 0.01)], k=len(x))

        # y_0, y_1 = x.clone(), x.clone()
        # y_0_pos, y_1_pos = x.clone(), x.clone()
        # y_0_neg, y_1_neg = x.clone(), x.clone()
        # meta_0, meta_1 = [], []
        # meta_0_neg, meta_1_neg = [], []

        # multiple positive samples
        num_positives = 3
        y_pos = [x.clone() for _ in range(num_positives)]
        meta_pos = []

        # multiple negative samples
        num_negatives = 10
        y_neg = [x.clone() for _ in range(num_negatives)]
        meta_neg = []

        for i in range(len(x)):
            ### Platform anomaly
            # m = [hist_sample(level_0_cdf, LEVEL_BINS), np.random.uniform(0, 0.5), hist_sample(length_0_cdf, LENGTH_BINS)]
            # y_0[i][0] = self.inject_platform(y_0[i][0], m[0], m[1], m[2])
            # meta_0.append(m)

            # First platform anomaly - original
            # m0 = [FIXED_LEVEL, FIXED_START, FIXED_LENGTH]
            m0 = [fixed_config_from_grid(CDF_LEVEL, GRID_LEVEL), 
                  random.choice(np.arange(0, 0.5, 0.01)),
                  fixed_config_from_grid(CDF_LENGTH, GRID_LENGTH)]
            # y_0[i][0] = self.inject_platform(y_0[i][0], *m0)
            # meta_0.append(m0)

            # generate positive samples
            for j in range(num_positives):
                pos_variation = [
                    m0[0] + np.random.uniform(-TAU, TAU),
                    max(m0[1] + np.random.uniform(-TAU, TAU), 0),
                    max(m0[2] + np.random.uniform(-TAU, TAU), 0)
                ]
                y_pos[j][i][0] = self.inject_platform(y_pos[j][i][0], *pos_variation)
            
            # generate negative samples
            for j in range(num_negatives):
                neg_variation = [
                    m0[0] + np.random.uniform(NEG_RANGE, -TAU) \
                        if np.random.random() > 0.5 else m0[0] + np.random.uniform(TAU, POS_RANGE),
                    max(m0[1] + np.random.uniform(NEG_RANGE, -TAU) \ 
                        if np.random.random() > 0.5 else m0[1] + np.random.uniform(TAU, POS_RANGE), 0),
                    max(m0[2] + np.random.uniform(NEG_RANGE, -TAU) \
                        if np.random.random() > 0.5 else m0[2] + np.random.uniform(TAU, POS_RANGE), 0)
                ]
                y_neg[j][i][0] = self.inject_platform(y_neg[j][i][0], *neg_variation)
            
        # concatenating samples
        all_samples = torch.cat([x] + y_pos + y_neg, dim = 0)
        outputs = self(all_samples)

        # meta_0, meta_0_neg = np.array(meta_0), np.array(meta_0_neg)
        # # meta_1, meta_1_neg = np.array(meta_1), np.array(meta_1_neg)
        # # meta_1, meta_1_neg = np.array(meta_1), np.array(meta_1_neg)

        # outputs = self(torch.cat([x, y_0, y_0_pos, y_0_neg, x_pos], dim=0))
        # outputs = self(torch.cat([x, y_0, y_0_pos, y_0_neg, y_1, y_1_pos, y_1_neg, x_pos], dim=0))
        # c_x, c_y_0, c_y_0_pos, c_y_0_neg, c_y_1, c_y_1_pos, c_y_1_neg, c_x_pos = torch.split(outputs, x.shape[0], dim=0)
        c_x, *c_y_pos, *c_y_neg = torch.split(outputs, [x.shape[0]] + [x.shape[0]]*num_positives + [x.shape[0]]*num_negatives, dim=0)
        #c_x, c_y_0, c_y_0_pos, c_y_0_neg, c_x_pos = torch.split(outputs, x.shape[0], dim=0)

        ### Anomalies should be close to the ones with the same type and similar hyperparameters, and far away from the ones with different types and normal.
        # loss_global_0 = self.info_loss(c_y_0, c_y_0_pos, torch.cat([c_x, c_x_pos, c_y_1, c_y_1_pos, c_y_1_neg], dim=0))
        # loss_global_1 = self.info_loss(c_y_1, c_y_1_pos, torch.cat([c_x, c_x_pos, c_y_0, c_y_0_pos, c_y_0_neg], dim=0))
        # loss_global = loss_global_0 + loss_global_1
        loss_global = sum(self.info_loss(c_x, c_y_p, torch.cat([c_x] + c_y_neg, dim=0)) for c_y_p in c_y_pos)

        ### Anomalies with far away hyperparameters should be far away propotional to delta.
        # loss_local_0 = hard_negative_loss(c_y_0, c_y_0_pos, c_y_0_neg, meta_0, meta_0_neg)
        # loss_local_1 = hard_negative_loss(c_y_1, c_y_1_pos, c_y_1_neg, meta_1, meta_1_neg)
        # loss_local = loss_local_0 + loss_local_1
        loss_local = sum(hard_negative_loss(c_x, c_y_p, torch.stack(c_y_neg), meta_pos[i], meta_neg) for i, c_y_p in enumerate(c_y_pos))

        ### Nomral should be close to each other, and far away from anomalies.
        # loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([c_y_0, c_y_0_pos, c_y_0_neg, c_y_1, c_y_1_pos, c_y_1_neg], dim=0))
        loss_normal = self.info_loss(c_x, c_x, torch.cat([torch.cat(c_y_pos, dim=0), torch.cat(c_y_neg, dim=0)], dim=0))

        loss = loss_global + loss_local + loss_normal
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, x, batch_idx):
        x_pos = self.normal_x[np.random.choice(len(self.normal_x), x.shape[0])]
        
        anomalies_start = random.choices([i for i in np.arange(0, 0.5, 0.01)], k=len(x))

        y_pos = [x.clone() for _ in range(3)]  # Multiple positive samples
        y_neg = [x.clone() for _ in range(10)]  # Multiple negative samples
        meta_pos = []
        meta_neg = []

        for i in range(len(x)):
            ### Platform anomaly
            # m = [hist_sample(level_0_cdf, LEVEL_BINS), np.random.uniform(0, 0.5), hist_sample(length_0_cdf, LENGTH_BINS)]
            # m0 = [FIXED_LEVEL, FIXED_START, FIXED_LENGTH]
            m0 = [fixed_config_from_grid(CDF_LEVEL, GRID_LEVEL), anomalies_start[i],
                  fixed_config_from_grid(CDF_LENGTH, GRID_LENGTH)]
            y_0 = x.clone()
            y_0[i][0] = self.inject_platform(y_0[i][0], *m0)

            # positive sample
            for j in range(3):
                pos_variation = [
                    m0[0] + np.random.uniform(-TAU, TAU),
                    max(m0[1] + np.random.uniform(-TAU, TAU), 0),
                    max(m0[2] + np.random.uniform(-TAU, TAU), 0)
                ]
                y_pos[j][i][0] = self.inject_platform(y_pos[j][i][0], *pos_variation)
                meta_pos.append(pos_variation)

            # negative sample
            for j in range(10):
                neg_variation = [
                    m0[0] + np.random.uniform(NEG_RANGE, -TAU) \
                        if np.random.random() > 0.5 else m0[0] + np.random.uniform(TAU, POS_RANGE),
                    max(m0[1] + np.random.uniform(NEG_RANGE, -TAU) \
                        if np.random.random() > 0.5 else m0[1] + np.random.uniform(TAU, POS_RANGE), 0),
                    max(m0[2] + np.random.uniform(NEG_RANGE, -TAU) \
                        if np.random.random() > 0.5 else m0[2] + np.random.uniform(TAU, POS_RANGE), 0)
                ]
                y_neg[j][i][0] = self.inject_platform(y_neg[j][i][0], *neg_variation)
                meta_neg.append(neg_variation)

        all_samples = torch.cat([x] + y_pos + y_neg + [x_pos], dim=0)
        outputs = self(all_samples)

        c_x, *c_y_pos, *c_y_neg, c_x_pos = torch.split(outputs, [x.shape[0]] + [x.shape[0]]*3 + [x.shape[0]]*10 + [x.shape[0]], dim=0)


        # loss_global_0 = self.info_loss(c_y_0, c_y_0_pos, torch.cat([c_x, c_x_pos, c_y_1, c_y_1_pos], dim=0))
        # loss_global_1 = self.info_loss(c_y_1, c_y_1_pos, torch.cat([c_x, c_x_pos, c_y_0, c_y_0_pos], dim=0))
        # loss_global = loss_global_0 + loss_global_1
        loss_global = sum(self.info_loss(c_x, c_y_p, torch.cat([c_x] + c_y_neg, dim=0)) for c_y_p in c_y_pos)

        # loss_local_0 = hard_negative_loss(c_y_0, c_y_0_pos, c_y_0_neg, meta_0, meta_0_neg)
        # loss_local_1 = hard_negative_loss(c_y_1, c_y_1_pos, c_y_1_neg, meta_1, meta_1_neg)
        # loss_local = loss_local_0 + loss_local_1
        loss_local = sum(hard_negative_loss(c_x, c_y_p, torch.stack(c_y_neg), meta_pos[i], meta_neg[i]) 
                     for i, c_y_p in enumerate(c_y_pos))

        # loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([c_y_0, c_y_0_pos, c_y_1, c_y_1_pos], dim=0))
        loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([torch.cat(c_y_pos, dim=0), torch.cat(c_y_neg, dim=0)], dim=0))

        loss = loss_global + loss_local + loss_normal
        self.log("loss_global", loss_global, prog_bar=True)
        self.log("loss_local", loss_local, prog_bar=True)
        self.log("loss_normal", loss_normal, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

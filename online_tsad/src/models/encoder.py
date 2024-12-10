import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from info_nce import InfoNCE

from .model import CNNEncoder


LENGTH_BINS = [0.2, 0.3, 0.4, 0.5]
LEVEL_BINS = [-1, -0.33, 0.33, 1]
TAU = 0.01
NEG_RANGE = -0.5
POS_RANGE = 0.5


def hist_sample(cdf, bins):
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    val = np.random.uniform(bins[bin_idx-1], bins[bin_idx])
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
        ts_row[start_a : start_a + length_a] = float(level)
        return ts_row

    def inject_mean(self, ts_row, level, start, length):
        start = int(len(ts_row) * start)
        length = int(len(ts_row) * length)
        ts_row[start : start + length] += float(level)
        return ts_row

    def training_step(self, x, batch_idx):
        if batch_idx not in self.normal_idx:
            self.normal_idx.add(batch_idx)
            self.normal_x = torch.cat([self.normal_x, x], dim=0)
        x_pos = self.normal_x[np.random.choice(len(self.normal_x), x.shape[0])]

        # a_config = self.a_config
        # level_0_cdf  = [0, a_config['level_0_h0'] , a_config['level_0_h0']  + a_config['level_0_h1'] , 1]
        # length_0_cdf = [0, a_config['length_0_h0'], a_config['length_0_h0'] + a_config['length_0_h1'], 1]
        # level_1_cdf  = [0, a_config['level_1_h0'] , a_config['level_1_h0']  + a_config['level_1_h1'] , 1]
        # length_1_cdf = [0, a_config['length_1_h0'], a_config['length_1_h0'] + a_config['length_1_h1'], 1]

        # y_0, y_1 = x.clone(), x.clone()
        # y_0_pos, y_1_pos = x.clone(), x.clone()
        # y_0_neg, y_1_neg = x.clone(), x.clone()
        # meta_0, meta_1, meta_0_neg, meta_1_neg = [], [], [], []

        # fixed parameters for platform anomaly - trial 1
        fixed_level = 0.5
        fixed_length = 0.3
        fixed_start = 0.2

        y_0 = x.clone()
        y_0_within, y_0_outside = x.clone(), x.clone()
        meta_0, meta_0_within, meta_0_outside = [], [], []

        for i in range(len(x)):
            ### Platform anomaly
            # m = [hist_sample(level_0_cdf, LEVEL_BINS), np.random.uniform(0, 0.5), hist_sample(length_0_cdf, LENGTH_BINS)]
            # y_0[i][0] = self.inject_platform(y_0[i][0], m[0], m[1], m[2])
            # meta_0.append(m)

            # platform anomaly fixed params
            m = [fixed_level, fixed_start, fixed_length]
            y_0[i][0] = self.inject_platform(y_0[i][0], fixed_level, fixed_start, fixed_length)
            meta_0.append([fixed_level, fixed_start, fixed_length])

            # Within epsilon neighborhood
            s0_within = fixed_level + np.random.uniform(low=-TAU, high=TAU)
            s1_within = max(fixed_start + np.random.uniform(low=-TAU, high=TAU), 0)
            s2_within = max(fixed_length + np.random.uniform(low=-TAU, high=TAU), 0)
            y_0_within[i][0] = self.inject_platform(y_0_within[i][0], s0_within, s1_within, s2_within)
            meta_0_within.append([s0_within, s1_within, s2_within])

            # Outside epsilon neighborhood
            s0_outside = fixed_level + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else fixed_level + np.random.uniform(low=TAU, high=POS_RANGE)
            s1_outside = max(fixed_start + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else fixed_start + np.random.uniform(low=TAU, high=POS_RANGE), 0)
            s2_outside = max(fixed_length + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else fixed_length + np.random.uniform(low=TAU, high=POS_RANGE), 0)
            y_0_outside[i][0] = self.inject_platform(y_0_outside[i][0], s0_outside, s1_outside, s2_outside)
            meta_0_outside.append([s0_outside, s1_outside, s2_outside])
    
            
            # ### Mean shift anomaly
            # m = [hist_sample(level_1_cdf, LEVEL_BINS), np.random.uniform(0, 0.5), hist_sample(length_1_cdf, LENGTH_BINS)]
            # m[0] = min(m[0], -0.1) if m[0] < 0 else max(m[0], 0.1)
            # y_1[i][0] = self.inject_mean(y_1[i][0], m[0], m[1], m[2])
            # meta_1.append(m)

            # s0 = m[0] + np.random.uniform(low=-TAU, high=TAU)
            # s1 = max(m[1] + np.random.uniform(low=-TAU, high=TAU), 0)
            # s2 = max(m[2] + np.random.uniform(low=-TAU, high=TAU), 0)
            # y_1_pos[i][0] = self.inject_mean(y_1_pos[i][0], s0, s1, s2)

            # s0 = m[0] + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else m[0] + np.random.uniform(low=TAU, high=POS_RANGE)
            # s1 = max(m[1] + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else m[1] + np.random.uniform(low=TAU, high=POS_RANGE), 0)
            # s2 = max(m[2] + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else m[2] + np.random.uniform(low=TAU, high=POS_RANGE), 0)
            # y_1_neg[i][0] = self.inject_mean(y_1_pos[i][0], s0, s1, s2)
            # meta_1_neg.append([s0, s1, s2])

        #meta_0, meta_0_neg = np.array(meta_0), np.array(meta_0_neg)
        #meta_1, meta_1_neg = np.array(meta_1), np.array(meta_1_neg)
        meta_0, meta_0_within, meta_0_outside = map(np.array, [meta_0, meta_0_within, meta_0_outside])


        outputs = self(torch.cat([x, y_0, y_0_within, y_0_outside, x_pos], dim=0))
        #outputs = self(torch.cat([x, y_0, y_0_pos, y_0_neg, x_pos], dim=0))
        #c_x, c_y_0, c_y_0_pos, c_y_0_neg, c_y_1, c_y_1_pos, c_y_1_neg, c_x_pos = torch.split(outputs, x.shape[0], dim=0)
        #c_x, c_y_0, c_y_0_pos, c_y_0_neg, c_x_pos = torch.split(outputs, x.shape[0], dim=0)
        c_x, c_y_0, c_y_0_within, c_y_0_outside, c_x_pos = torch.split(outputs, x.shape[0], dim=0)

        ### Anomalies should be close to the ones with the same type and similar hyperparameters, and far away from the ones with different types and normal.
        # loss_global_0 = self.info_loss(c_y_0, c_y_0_pos, torch.cat([c_x, c_x_pos, c_y_1, c_y_1_pos, c_y_1_neg], dim=0))
        # loss_global_1 = self.info_loss(c_y_1, c_y_1_pos, torch.cat([c_x, c_x_pos, c_y_0, c_y_0_pos, c_y_0_neg], dim=0))
        # loss_global = loss_global_0 + loss_global_1
        #loss_global = self.info_loss(c_y_0, c_y_0_pos, torch.cat([c_x, c_x_pos], dim=0))
        loss_global = self.info_loss(c_y_0, c_y_0_within, torch.cat([c_x, c_x_pos, c_y_0_outside], dim=0))

        ### Anomalies with far away hyperparameters should be far away propotional to delta.
        # loss_local_0 = hard_negative_loss(c_y_0, c_y_0_pos, c_y_0_neg, meta_0, meta_0_neg)
        # loss_local_1 = hard_negative_loss(c_y_1, c_y_1_pos, c_y_1_neg, meta_1, meta_1_neg)
        # loss_local = loss_local_0 + loss_local_1
        #loss_local = hard_negative_loss(c_y_0, c_y_0_pos, c_y_0_neg, meta_0, meta_0_neg)
        loss_local = hard_negative_loss(c_y_0, c_y_0_within, c_y_0_outside, meta_0, meta_0_outside)

        
        ### Nomral should be close to each other, and far away from anomalies.
        #loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([c_y_0, c_y_0_pos, c_y_0_neg, c_y_1, c_y_1_pos, c_y_1_neg], dim=0))
        #loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([c_y_0, c_y_0_pos, c_y_0_neg], dim=0))
        loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([c_y_0, c_y_0_within, c_y_0_outside], dim=0))

        loss = loss_global + loss_local + loss_normal
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, x, batch_idx):
        x_pos = self.normal_x[np.random.choice(len(self.normal_x), x.shape[0])]

        # a_config = self.a_config
        # level_0_cdf  = [0, a_config['level_0_h0'] , a_config['level_0_h0']  + a_config['level_0_h1'] , 1]
        # length_0_cdf = [0, a_config['length_0_h0'], a_config['length_0_h0'] + a_config['length_0_h1'], 1]
        # level_1_cdf  = [0, a_config['level_1_h0'] , a_config['level_1_h0']  + a_config['level_1_h1'] , 1]
        # length_1_cdf = [0, a_config['length_1_h0'], a_config['length_1_h0'] + a_config['length_1_h1'], 1]

        # y_0, y_1 = x.clone(), x.clone()
        # y_0_pos, y_1_pos = x.clone(), x.clone()
        # y_0_neg, y_1_neg = x.clone(), x.clone()
        # meta_0, meta_1, meta_0_neg, meta_1_neg = [], [], [], []

        # fixed parameters for platform anomaly - trial 1
        fixed_level = 0.5
        fixed_length = 0.3
        fixed_start = 0.2

        y_0 = x.clone()
        y_0_within, y_0_outside = x.clone(), x.clone()
        meta_0, meta_0_within, meta_0_outside = [], [], []

        for i in range(len(x)):
            ### Platform anomaly
            #m = [hist_sample(level_0_cdf, LEVEL_BINS), np.random.uniform(0, 0.5), hist_sample(length_0_cdf, LENGTH_BINS)]
            m = [fixed_level, fixed_start, fixed_length]
            y_0[i][0] = self.inject_platform(y_0[i][0], m[0], m[1], m[2])
            meta_0.append(m)

            # Within epsilon neighborhood
            s0_within = fixed_level + np.random.uniform(low=-TAU, high=TAU)
            s1_within = max(fixed_start + np.random.uniform(low=-TAU, high=TAU), 0)
            s2_within = max(fixed_length + np.random.uniform(low=-TAU, high=TAU), 0)
            y_0_within[i][0] = self.inject_platform(y_0_within[i][0], s0_within, s1_within, s2_within)
            meta_0_within.append([s0_within, s1_within, s2_within])
            
            # Outside epsilon neighborhood
            s0_outside = fixed_level + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else fixed_level + np.random.uniform(low=TAU, high=POS_RANGE)
            s1_outside = max(fixed_start + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else fixed_start + np.random.uniform(low=TAU, high=POS_RANGE), 0)
            s2_outside = max(fixed_length + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else fixed_length + np.random.uniform(low=TAU, high=POS_RANGE), 0)
            y_0_outside[i][0] = self.inject_platform(y_0_outside[i][0], s0_outside, s1_outside, s2_outside)
            meta_0_outside.append([s0_outside, s1_outside, s2_outside])
                
            # ### Mean shift anomaly
            # m = [hist_sample(level_1_cdf, LEVEL_BINS), np.random.uniform(0, 0.5), hist_sample(length_1_cdf, LENGTH_BINS)]
            # m[0] = min(m[0], -0.1) if m[0] < 0 else max(m[0], 0.1)
            # y_1[i][0] = self.inject_mean(y_1[i][0], m[0], m[1], m[2])
            # meta_1.append(m)

            # s0 = m[0] + np.random.uniform(low=-TAU, high=TAU)
            # s1 = max(m[1] + np.random.uniform(low=-TAU, high=TAU), 0)
            # s2 = max(m[2] + np.random.uniform(low=-TAU, high=TAU), 0)
            # y_1_pos[i][0] = self.inject_mean(y_1_pos[i][0], s0, s1, s2)

            # s0 = m[0] + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else m[0] + np.random.uniform(low=TAU, high=POS_RANGE)
            # s1 = max(m[1] + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else m[1] + np.random.uniform(low=TAU, high=POS_RANGE), 0)
            # s2 = max(m[2] + np.random.uniform(low=NEG_RANGE, high=-TAU) if np.random.random() > 0.5 else m[2] + np.random.uniform(low=TAU, high=POS_RANGE), 0)
            # y_1_neg[i][0] = self.inject_mean(y_1_pos[i][0], s0, s1, s2)
            # meta_1_neg.append([s0, s1, s2])

        #meta_0, meta_0_neg = np.array(meta_0), np.array(meta_0_neg)
       #meta_1, meta_1_neg = np.array(meta_1), np.array(meta_1_neg)
        meta_0, meta_0_within, meta_0_outside = map(np.array, [meta_0, meta_0_within, meta_0_outside])

        # outputs = self(torch.cat([x, y_0, y_0_pos, y_0_neg, y_1, y_1_pos, y_1_neg, x_pos], dim=0))
        # c_x, c_y_0, c_y_0_pos, c_y_0_neg, c_y_1, c_y_1_pos, c_y_1_neg, c_x_pos = torch.split(outputs, x.shape[0], dim=0)
        # outputs = self(torch.cat([x, y_0, y_0_pos, y_0_neg, x_pos], dim=0))
        # c_x, c_y_0, c_y_0_pos, c_y_0_neg, c_x_pos = torch.split(outputs, x.shape[0], dim=0)
        outputs = self(torch.cat([x, y_0, y_0_within, y_0_outside, x_pos], dim=0))
        c_x, c_y_0, c_y_0_within, c_y_0_outside, c_x_pos = torch.split(outputs, x.shape[0], dim=0)
     
        # loss_global_0 = self.info_loss(c_y_0, c_y_0_pos, torch.cat([c_x, c_x_pos, c_y_1, c_y_1_pos], dim=0))
        # loss_global_1 = self.info_loss(c_y_1, c_y_1_pos, torch.cat([c_x, c_x_pos, c_y_0, c_y_0_pos], dim=0))
        #loss_global = loss_global_0 + loss_global_1
        #loss_global = self.info_loss(c_y_0, c_y_0_pos, torch.cat([c_x, c_x_pos], dim=0))
        loss_global = self.info_loss(c_y_0, c_y_0_within, torch.cat([c_x, c_x_pos, c_y_0_outside], dim=0))


        # loss_local_0 = hard_negative_loss(c_y_0, c_y_0_pos, c_y_0_neg, meta_0, meta_0_neg)
        # loss_local_1 = hard_negative_loss(c_y_1, c_y_1_pos, c_y_1_neg, meta_1, meta_1_neg)
        # loss_local = loss_local_0 + loss_local_1
        #loss_local = hard_negative_loss(c_y_0, c_y_0_pos, c_y_0_neg, meta_0, meta_0_neg)
        loss_local = hard_negative_loss(c_y_0, c_y_0_within, c_y_0_outside, meta_0, meta_0_outside)
        
        #loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([c_y_0, c_y_0_pos, c_y_1, c_y_1_pos], dim=0))
        #loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([c_y_0, c_y_0_pos], dim=0))
        loss_normal = self.info_loss(c_x, c_x_pos, torch.cat([c_y_0, c_y_0_within, c_y_0_outside], dim=0))

        loss = loss_global + loss_local + loss_normal
        self.log("loss_global", loss_global, prog_bar=True)
        self.log("loss_local", loss_local, prog_bar=True)
        self.log("loss_normal", loss_normal, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
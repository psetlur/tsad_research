import numpy as np
import torch
import pytorch_lightning as pl
from info_nce import InfoNCE
from .model import CNNEncoder
from scipy.stats import truncnorm, bernoulli

LENGTH_BINS = [0.2, 0.3, 0.4, 0.5]
LEVEL_BINS = [-1, -0.33, 0.33, 1]
TAU = 0.01
TAU_LEVEL = 0.01
TAU_LENGTH = 0.001
TAU_P = 0.01
RANGE = 0.5
RANGE_LEVEL = 0.5
RANGE_LENGTH = 0.1
RANGE_P = 0.1

FIXED_LEVEL = 0.5
FIXED_LENGTH = 0.3

GRID_PLATFORM_LEVEL = np.round(np.arange(-1, 1.1, 0.1), 1)
GRID_MEAN_LEVEL = np.round(np.arange(-1, 1.1, 0.1), 1)
GRID_MEAN_LEVEL = GRID_MEAN_LEVEL[GRID_MEAN_LEVEL != 0]
GRID_LENGTH = np.round(np.arange(0.2, 0.51, 0.01), 2)
GRID_P = np.round(np.arange(0.1, 1.1, 0.1), 1)
CDF_PLATFORM_LEVEL = np.arange(0, 1, 1 / len(GRID_PLATFORM_LEVEL))
CDF_MEAN_LEVEL = np.arange(0, 1, 1 / len(GRID_PLATFORM_LEVEL))
CDF_LENGTH = np.arange(0, 1, 1 / len(GRID_LENGTH))
CDF_P = np.arange(0, 1, 1 / len(GRID_P))

NUM_POSITIVE = 3
NUM_NEGATIVE = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def config_from_grid(cdf, grid):
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    return grid[bin_idx - 1]


class Encoder(pl.LightningModule):
    def __init__(self, args, ts_input_size, lr):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        self.encoder = CNNEncoder(ts_input_size)
        self.lr = lr
        self.temperature = 0.1
        self.info_loss = InfoNCE(negative_mode='unpaired', temperature=self.temperature)

        self.normal_idx = set()
        self.normal_x = torch.tensor([]).to(device)

        if self.args.trail == 'second_anomaly':
            self.anomaly_types = ['platform', 'mean']
        elif self.args.trail in ['inject_spike', 'inject_spike_with_one']:
            self.anomaly_types = ['spike']
        elif self.args.trail == 'three_anomalies':
            self.anomaly_types = ['platform', 'mean', 'spike']
        else:
            self.anomaly_types = ['platform']
            # raise Exception('Unsupported trail.')

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
        # ts_row = ((ts_row - ts_row.min()) / (ts_row.max() - ts_row.min())) * 2 - 1
        return ts_row

    def inject_spike(self, ts_row, p):
        direction = np.random.rand(len(ts_row)) < 0.5
        spikes = np.zeros(len(ts_row))
        spikes[direction] = truncnorm.rvs(-np.inf, -2, loc=0, scale=1, size=direction.sum())
        spikes[~direction] = truncnorm.rvs(2, np.inf, loc=0, scale=1, size=(~direction).sum())
        mask = bernoulli.rvs(p=p, size=len(ts_row))
        ts_row += torch.tensor(mask * spikes).to(ts_row.device)
        return ts_row

    def inject(self, anomaly_type, ts, config):
        if anomaly_type == 'platform':
            return self.inject_platform(ts, *config)
        elif anomaly_type == 'mean':
            return self.inject_mean(ts, *config)
        elif anomaly_type == 'spike':
            return self.inject_spike(ts, *config)
        else:
            raise Exception('Unsupported anomaly_type.')

    def main_process(self, x, process):
        x_pos = self.normal_x[np.random.choice(len(self.normal_x), x.shape[0])]
        c_x = self(x)
        c_x_pos = self(x_pos)
        c_y_dict, c_y_pos_dict, c_y_neg_dict, meta_dict, meta_neg_dict = dict(), dict(), dict(), dict(), dict()
        for anomaly_type in self.anomaly_types:
            y, y_pos = x.clone(), x.clone()
            meta, meta_pos = list(), list()
            y_neg = [x.clone() for _ in range(NUM_NEGATIVE)]
            meta_neg = [list() for _ in range(NUM_NEGATIVE)]

            for i in range(len(x)):
                if anomaly_type == 'spike':
                    m = [config_from_grid(CDF_P, GRID_P)]
                elif anomaly_type == 'platform':
                    m = [config_from_grid(CDF_PLATFORM_LEVEL, GRID_PLATFORM_LEVEL), np.random.uniform(0, 0.5),
                         config_from_grid(CDF_LENGTH, GRID_LENGTH)]
                else:  # anomaly_type == 'mean'
                    m = [config_from_grid(CDF_MEAN_LEVEL, GRID_MEAN_LEVEL), np.random.uniform(0, 0.5),
                         config_from_grid(CDF_LENGTH, GRID_LENGTH)]
                y[i][0] = self.inject(anomaly_type=anomaly_type, ts=y[i][0], config=m)
                meta.append(m)

                # positive samples
                if anomaly_type == 'spike':
                    m_pos = [m[0] + np.random.uniform(low=-TAU_P, high=TAU_P)]
                    m_pos[0] = min(m_pos[0], 1)
                else:
                    s0 = m[0] + np.random.uniform(low=-TAU_LEVEL, high=TAU_LEVEL)
                    s2 = max(m[2] + np.random.uniform(low=-TAU_LENGTH, high=TAU_LENGTH), 0)
                    m_pos = [s0, np.random.uniform(0, 0.5), s2]
                y_pos[i][0] = self.inject(anomaly_type=anomaly_type, ts=y_pos[i][0], config=m_pos)
                meta_pos.append(m_pos)

                # negative samples
                for neg_index in range(NUM_NEGATIVE):
                    if anomaly_type == 'spike':
                        m_neg = [m[0] + np.random.uniform(low=-RANGE_P, high=-TAU_P)
                                 if np.random.random() > 0.5 else m[0] + np.random.uniform(low=TAU_P, high=RANGE_P)]
                        m_neg[0] = max(m_neg[0], 0.1)
                        m_neg[0] = min(m_neg[0], 1)
                    else:
                        s0_neg = m[0] + np.random.uniform(low=-RANGE_LEVEL, high=-TAU_LEVEL) \
                            if np.random.random() > 0.5 else m[0] + np.random.uniform(low=TAU_LEVEL, high=RANGE_LEVEL)
                        s2_neg = max(m[2] + np.random.uniform(low=-RANGE_LENGTH, high=-TAU_LENGTH)
                                     if np.random.random() > 0.5 else
                                     m[2] + np.random.uniform(low=TAU_LENGTH, high=RANGE_LENGTH), 0)
                        m_neg = [s0_neg, np.random.uniform(0, 0.5), s2_neg]
                    y_neg[neg_index][i][0] = self.inject(anomaly_type=anomaly_type, ts=y_neg[neg_index][i][0],
                                                         config=m_neg)
                    meta_neg[neg_index].append(m_neg)

            c_y_dict[anomaly_type] = self(y)
            c_y_pos_dict[anomaly_type] = self(y_pos)
            c_y_neg_dict[anomaly_type] = [self(y_neg[i]) for i in range(NUM_NEGATIVE)]
            meta_dict[anomaly_type] = meta
            meta_neg_dict[anomaly_type] = meta_neg

        ### Anomalies should be close to the ones with the same type and similar hyperparameters, and far away from
        # the ones with different types and normal.
        loss_global = 0
        for anomaly_type in self.anomaly_types:
            if len(self.anomaly_types) == 1:
                loss_global += self.info_loss(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type],
                                              torch.cat([c_x, c_x_pos], dim=0))
            else:
                _c_y = list()
                for _anomaly_type in self.anomaly_types:
                    if _anomaly_type != anomaly_type:
                        _c_y.append(c_y_dict[_anomaly_type])
                _c_y = torch.cat(_c_y, dim=0)
                loss_global += self.info_loss(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type],
                                              torch.cat([c_x, c_x_pos, _c_y], dim=0))
        loss_global /= len(self.anomaly_types)

        ### Anomalies with far away hyperparameters should be far away propotional to delta.
        loss_local = 0
        for anomaly_type in self.anomaly_types:
            loss_local += sum([hard_negative_loss(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type],
                                                  c_y_neg_dict[anomaly_type][i],
                                                  np.array(meta_dict[anomaly_type]),
                                                  np.array(meta_neg_dict[anomaly_type][i]))
                               for i in range(NUM_NEGATIVE)]) / NUM_NEGATIVE
        loss_local /= len(self.anomaly_types)

        ### Nomral should be close to each other, and far away from anomalies.
        c_y_neg_dict = [torch.cat(c_y_neg_dict[anomaly_type], dim=0) for anomaly_type in self.anomaly_types]
        loss_normal = self.info_loss(c_x, c_x_pos, torch.cat(
            [torch.cat(list(c_y_dict.values()), dim=0), torch.cat(list(c_y_pos_dict.values()), dim=0),
             torch.cat(c_y_neg_dict, dim=0)], dim=0))

        loss = loss_global + loss_local + loss_normal

        if loss_global > 0 and loss_local > 0 and loss_normal > 0:
            pass
        else:
            # raise Exception(f'loss_global: {loss_global}, loss_local: {loss_local}, loss_normal: {loss_normal}')
            raise Warning(f'loss_global: {loss_global}, loss_local: {loss_local}, loss_normal: {loss_normal}')

        self.log("loss_global", loss_global, prog_bar=True)
        self.log("loss_local", loss_local, prog_bar=True)
        self.log("loss_normal", loss_normal, prog_bar=True)
        self.log(f"{process}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, x, batch_idx):
        self.normal_x = self.normal_x.to(device)
        if batch_idx not in self.normal_idx:
            self.normal_idx.add(batch_idx)
            self.normal_x = torch.cat([self.normal_x, x], dim=0).to(device)
        return self.main_process(x=x, process='train')

    def validation_step(self, x, batch_idx):
        return self.main_process(x=x, process='val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

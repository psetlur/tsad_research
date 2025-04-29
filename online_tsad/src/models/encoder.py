import numpy as np
import torch
import pytorch_lightning as pl
from info_nce import InfoNCE
from .model import CNNEncoder
import torch.nn.functional as F
import random


def generate_parameters(min, max, step, tau, exclude_zero=False):
    grid = np.arange(min, max + step, step)
    if exclude_zero:
        grid = grid[grid != 0]
    cdf = np.arange(0, 1, 1 / len(grid))
    return {'min': min, 'max': max, 'grid': grid, 'cdf': cdf, 'tau': tau}


CONFIGS = {'platform': {'level': {'min': -1, 'max': 1, 'step': 0.1, 'tau': 0.01},
                        'length': {'min': 100, 'max': 250, 'step': 10, 'tau': 1}},
           'mean': {'level': {'min': -1, 'max': 1, 'step': 0.1, 'tau': 0.01, 'exclude_zero': True},
                    'length': {'min': 100, 'max': 250, 'step': 10, 'tau': 1}},
           'spike': {'level': {'min': 2, 'max': 20, 'step': 2, 'tau': 0.2}},
           'amplitude': {'level': [{'min': 0.1, 'max': 0.9, 'step': 0.1, 'tau': 0.01},
                                   {'min': 2, 'max': 10, 'step': 1, 'tau': 0.1}],
                         'length': {'min': 100, 'max': 250, 'step': 10, 'tau': 1}},
           'trend': {'slope': {'min': -0.01, 'max': 0.01, 'step': 0.001, 'tau': 0.0001, 'exclude_zero': True},
                     'length': {'min': 100, 'max': 250, 'step': 10, 'tau': 1}},
           'variance': {'level': {'min': 0.1, 'max': 0.5, 'step': 0.05, 'tau': 0.005},
                        'length': {'min': 100, 'max': 250, 'step': 10, 'tau': 1}}}

PARAMETERS = {}
for anomaly_type, params in CONFIGS.items():
    PARAMETERS[anomaly_type] = {}
    for key, config in params.items():
        if isinstance(config, list):
            PARAMETERS[anomaly_type][key] = [generate_parameters(**cfg) for cfg in config]
        else:
            PARAMETERS[anomaly_type][key] = generate_parameters(**config)

LENGTH_BINS = [0.2, 0.3, 0.4, 0.5]
LEVEL_BINS = [-1, -0.33, 0.33, 1]

NUM_POSITIVE = 1
NUM_NEGATIVE = 10


def hist_sample(cdf, bins):
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    val = np.random.uniform(bins[bin_idx - 1], bins[bin_idx])
    return val


def local(z_anc, z_pos, z_neg, temperature=0.1):
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    # meta_anc = torch.tensor(meta_anc, dtype=torch.float).to(z_anc.device)
    # meta_neg = torch.tensor(meta_neg, dtype=torch.float).to(z_anc.device)
    # meta_diff = torch.abs(meta_neg.permute(1, 0, 2) - meta_anc.unsqueeze(1)).mean()
    z_anc, z_pos, z_neg = normalize(z_anc, z_pos, z_neg)
    positive_logit = torch.sum(z_anc * z_pos, dim=1, keepdim=True)
    # negative_logits = torch.bmm(z_anc.unsqueeze(1), z_neg.permute(1, 0, 2).transpose(1, 2)).squeeze(1) / (1 +
    # meta_diff)
    negative_logits = torch.bmm(z_anc.unsqueeze(1), z_neg.permute(1, 0, 2).transpose(1, 2)).squeeze(1)
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long, device=z_anc.device)
    return F.cross_entropy(logits / temperature, labels, reduction='mean')


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
        self.info_loss = InfoNCE()
        self.normal_idx = set()
        self.normal_x = torch.tensor([]).to(self.args.device)
        self.anomaly_types = ['platform', 'mean', 'spike', 'amplitude', 'trend', 'variance']
        # self.anomaly_types = ['platform']
        # self.anomaly_types = ['spike']

    def forward(self, x):
        x = self.encoder(x)
        return x

    def inject_platform(self, ts_row, level, length, start):
        start_a = int(len(ts_row) * start)
        ts_row[start_a: start_a + length] = float(level)
        return ts_row

    def inject_mean(self, ts_row, level, length, start):
        start_a = int(len(ts_row) * start)
        ts_row[start_a: start_a + length] += float(level)
        return ts_row

    def inject_spike(self, ts_row, level, start):
        start_a = int(len(ts_row) * start)
        ts_row[start_a] = ts_row[start_a] + level if np.random.rand() < 0.5 else ts_row[start_a] - level
        return ts_row

    def inject_amplitude(self, ts_row, level, length, start):
        start_a = int(len(ts_row) * start)
        amplitude_bell = torch.tensor(np.ones(length) * level, device=ts_row.device)
        ts_row[start_a: start_a + length] *= amplitude_bell
        return ts_row

    def inject_trend(self, ts_row, slope, length, start):
        start_a = int(len(ts_row) * start)
        slope_a = torch.tensor(np.arange(0, length) * slope, device=ts_row.device)
        ts_row[start_a: start_a + length] += slope_a
        ts_row[start_a + length:] += torch.tensor(np.full(len(ts_row) - start_a - length, slope_a[-1].cpu().item()),
                                                    device=ts_row.device)
        return ts_row

    def inject_variance(self, ts_row, level, length, start):
        start_a = int(len(ts_row) * start)
        var = torch.tensor(np.random.normal(0, level, length), device=ts_row.device)
        ts_row[start_a: start_a + length] += var
        return ts_row

    def main_process(self, x, process):
        x_pos = self.normal_x[np.random.choice(len(self.normal_x), x.shape[0])]
        c_x = self(x)
        c_x_pos = self(x_pos)
        c_y_dict, c_y_pos_dict, c_y_neg_dict = dict(), dict(), dict()
        for anomaly_type in self.anomaly_types:
            y = x.clone()
            y_pos = [x.clone() for _ in range(NUM_POSITIVE)]
            y_neg = [x.clone() for _ in range(NUM_NEGATIVE)]
            inject = getattr(self, f"inject_{anomaly_type}")
            for i in range(len(x)):
                m = list()
                start = np.random.uniform(0, 1)
                index = 0 if np.random.random() < 0.5 else 1
                for config in PARAMETERS[anomaly_type].keys():
                    if type(PARAMETERS[anomaly_type][config]) is list:
                        m.append(config_from_grid(PARAMETERS[anomaly_type][config][index]['cdf'],
                                                  PARAMETERS[anomaly_type][config][index]['grid']))
                    else:
                        m.append(config_from_grid(PARAMETERS[anomaly_type][config]['cdf'],
                                                  PARAMETERS[anomaly_type][config]['grid']))
                    if config == 'length':
                        start = np.random.uniform(0, 0.5)
                y[i][0] = inject(y[i][0], *m, start=start)

                # inject positive
                for pos_index in range(NUM_POSITIVE):
                    m_pos = list()
                    start = np.random.uniform(0, 1)
                    for j, config in enumerate(PARAMETERS[anomaly_type].keys()):
                        if config == 'length':
                            start = np.random.uniform(0, 0.5)
                            m_pos.append(np.random.choice([m[j] - PARAMETERS[anomaly_type][config]['tau'], m[j],
                                                           m[j] + PARAMETERS[anomaly_type][config]['tau']]))
                            if np.abs(m[j] - m_pos[j]) > PARAMETERS[anomaly_type][config]['tau']:
                                raise Exception()
                        else:
                            if type(PARAMETERS[anomaly_type][config]) is list:
                                m_pos.append(
                                    np.random.uniform(max(m[j] - PARAMETERS[anomaly_type][config][index]['tau'],
                                                          PARAMETERS[anomaly_type][config][index]['min']),
                                                      min(m[j] + PARAMETERS[anomaly_type][config][index]['tau'],
                                                          PARAMETERS[anomaly_type][config][index]['max'])))
                                if np.abs(m[j] - m_pos[j]) > PARAMETERS[anomaly_type][config][index]['tau']:
                                    raise Exception()
                            else:
                                m_pos.append(np.random.uniform(max(m[j] - PARAMETERS[anomaly_type][config]['tau'],
                                                                   PARAMETERS[anomaly_type][config]['min']),
                                                               min(m[j] + PARAMETERS[anomaly_type][config]['tau'],
                                                                   PARAMETERS[anomaly_type][config]['max'])))
                                if np.abs(m[j] - m_pos[j]) > PARAMETERS[anomaly_type][config]['tau']:
                                    raise Exception()
                    y_pos[pos_index][i][0] = inject(y_pos[pos_index][i][0], *m_pos, start=start)

                # inject negative
                for neg_index in range(NUM_NEGATIVE):
                    m_neg = list()
                    start = np.random.uniform(0, 1)
                    for j, config in enumerate(PARAMETERS[anomaly_type].keys()):
                        if config == 'length':
                            start = np.random.uniform(0, 0.5)
                            m_neg.append(np.random.randint(PARAMETERS[anomaly_type][config]['min'],
                                                           m[j] - PARAMETERS[anomaly_type][config]['tau'] - 1)
                                         if np.random.random() < ((m[j] - PARAMETERS[anomaly_type][config]['min'] -
                                                                   PARAMETERS[anomaly_type][config]['tau'] - 1) / (
                                                                          PARAMETERS[anomaly_type][config]['max'] -
                                                                          PARAMETERS[anomaly_type][config]['min'] - 2 *
                                                                          PARAMETERS[anomaly_type][config]['tau'] - 1))
                                         else np.random.randint(m[j] + PARAMETERS[anomaly_type][config]['tau'] + 1,
                                                                PARAMETERS[anomaly_type][config]['max']))
                            if np.abs(m[j] - m_neg[j]) <= PARAMETERS[anomaly_type][config]['tau']:
                                raise Exception()
                        else:
                            if type(PARAMETERS[anomaly_type][config]) is list:
                                m_neg.append(np.random.uniform(
                                    PARAMETERS[anomaly_type][config][index]['min'],
                                    m[j] - PARAMETERS[anomaly_type][config][index]['tau'])
                                             if np.random.random() < (
                                        (m[j] - PARAMETERS[anomaly_type][config][index]['min'] -
                                         PARAMETERS[anomaly_type][config][index]['tau']) / (
                                                PARAMETERS[anomaly_type][config][index]['max'] -
                                                PARAMETERS[anomaly_type][config][index]['min'] - 2 *
                                                PARAMETERS[anomaly_type][config][index]['tau']))
                                             else np.random.uniform(
                                    m[j] + PARAMETERS[anomaly_type][config][index]['tau'],
                                    PARAMETERS[anomaly_type][config][index]['max']))
                                if np.abs(m[j] - m_neg[j]) <= PARAMETERS[anomaly_type][config][index]['tau']:
                                    raise Exception()
                            else:
                                m_neg.append(np.random.uniform(
                                    PARAMETERS[anomaly_type][config]['min'],
                                    m[j] - PARAMETERS[anomaly_type][config]['tau'])
                                             if np.random.random() < ((m[j] - PARAMETERS[anomaly_type][config]['min'] -
                                                                       PARAMETERS[anomaly_type][config]['tau']) / (
                                                                              PARAMETERS[anomaly_type][config]['max'] -
                                                                              PARAMETERS[anomaly_type][config][
                                                                                  'min'] - 2 *
                                                                              PARAMETERS[anomaly_type][config]['tau']))
                                             else np.random.uniform(m[j] + PARAMETERS[anomaly_type][config]['tau'],
                                                                    PARAMETERS[anomaly_type][config]['max']))
                                if np.abs(m[j] - m_neg[j]) <= PARAMETERS[anomaly_type][config]['tau']:
                                    raise Exception()
                    y_neg[neg_index][i][0] = inject(y_neg[neg_index][i][0], *m_neg, start=start)

            c_y_dict[anomaly_type] = self(y)
            c_y_pos_dict[anomaly_type] = [self(y_pos[i]) for i in range(NUM_POSITIVE)]
            c_y_neg_dict[anomaly_type] = [self(y_neg[i]) for i in range(NUM_NEGATIVE)]

        # Anomalies should be close to the ones with the same type and similar hyperparameters, and far away from the
        # ones with different types and normal.
        loss_global = 0
        for anomaly_type in self.anomaly_types:
            c_y_others = list()
            for _anomaly_type in self.anomaly_types:
                if _anomaly_type != anomaly_type:
                    c_y_others.append(c_y_dict[_anomaly_type])
            if len(c_y_others) == 0:
                loss_global += sum([self.info_loss(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type][i],
                                                   torch.cat([c_x, c_x_pos], dim=0))
                                    for i in range(NUM_POSITIVE)]) / NUM_POSITIVE
            else:
                c_y_others = torch.cat(c_y_others, dim=0)
                loss_global += sum([self.info_loss(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type][i],
                                                   torch.cat([c_x, c_x_pos, c_y_others], dim=0))
                                    for i in range(NUM_POSITIVE)]) / NUM_POSITIVE
        loss_global /= len(self.anomaly_types)

        # Anomalies with far away hyperparameters should be far away propotional to delta.
        loss_local = 0
        for anomaly_type in self.anomaly_types:
            loss_local += sum([local(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type][j],
                                     torch.stack(c_y_neg_dict[anomaly_type], dim=0))
                               for j in range(NUM_POSITIVE)]) / NUM_POSITIVE
        loss_local /= len(self.anomaly_types)

        # Nomral should be close to each other, and far away from anomalies.
        c_y_pos = [torch.cat(c_y_pos_dict[anomaly_type], dim=0) for anomaly_type in self.anomaly_types]
        c_y_neg = [torch.cat(c_y_neg_dict[anomaly_type], dim=0) for anomaly_type in self.anomaly_types]
        loss_normal = self.info_loss(c_x, c_x_pos, torch.cat(
            [torch.cat(list(c_y_dict.values()), dim=0), torch.cat(c_y_pos, dim=0), torch.cat(c_y_neg, dim=0)], dim=0))

        loss = loss_global + loss_local + loss_normal

        if loss_global > 0 and loss_local > 0 and loss_normal > 0:
            pass
        else:
            raise Exception(f'loss_global: {loss_global}, loss_local: {loss_local}, loss_normal: {loss_normal}')

        self.log("loss_global", loss_global, prog_bar=True)
        self.log("loss_local", loss_local, prog_bar=True)
        self.log("loss_normal", loss_normal, prog_bar=True)
        self.log(f"{process}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, x, batch_idx):
        self.normal_x = self.normal_x.to(self.args.device)
        if batch_idx not in self.normal_idx:
            self.normal_idx.add(batch_idx)
            self.normal_x = torch.cat([self.normal_x, x], dim=0).to(self.args.device)
        return self.main_process(x=x, process='train')

    def validation_step(self, x, batch_idx):
        return self.main_process(x=x, process='val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

import numpy as np
import torch
import pytorch_lightning as pl
from info_nce import InfoNCE
from .model import CNNEncoder
from scipy.stats import bernoulli

MIN_PLATFORM_LEVEL = -1
MIN_PLATFORM_LENGTH = 0.2
MAX_PLATFORM_LEVEL = 1
MAX_PLATFORM_LENGTH = 0.5
PLATFORM_LEVEL_STEP = 0.1
PLATFORM_LENGTH_STEP = 0.01
RANGE_PLATFORM_LEVEL = 0.5
RANGE_PLATFORM_LENGTH = 0.1
GRID_PLATFORM_LEVEL = np.arange(MIN_PLATFORM_LEVEL, MAX_PLATFORM_LEVEL + PLATFORM_LEVEL_STEP, PLATFORM_LEVEL_STEP)
GRID_PLATFORM_LENGTH = np.arange(MIN_PLATFORM_LENGTH, MAX_PLATFORM_LENGTH + PLATFORM_LENGTH_STEP, PLATFORM_LENGTH_STEP)
CDF_PLATFORM_LEVEL = np.arange(0, 1, 1 / len(GRID_PLATFORM_LEVEL))
CDF_PLATFORM_LENGTH = np.arange(0, 1, 1 / len(GRID_PLATFORM_LENGTH))
TAU_PLATFORM_LEVEL = 0.01
TAU_PLATFORM_LENGTH = 0.001

MIN_MEAN_LEVEL = -1
MIN_MEAN_LENGTH = 0.2
MAX_MEAN_LEVEL = 1
MAX_MEAN_LENGTH = 0.5
MEAN_LEVEL_STEP = 0.1
MEAN_LENGTH_STEP = 0.01
RANGE_MEAN_LEVEL = 0.5
RANGE_MEAN_LENGTH = 0.1
GRID_MEAN_LEVEL = np.arange(MIN_MEAN_LEVEL, MAX_MEAN_LEVEL + MEAN_LEVEL_STEP, MEAN_LEVEL_STEP)
GRID_MEAN_LEVEL = GRID_MEAN_LEVEL[GRID_MEAN_LEVEL != 0]
GRID_MEAN_LENGTH = np.arange(MIN_MEAN_LENGTH, MAX_MEAN_LENGTH + MEAN_LENGTH_STEP, MEAN_LENGTH_STEP)
CDF_MEAN_LEVEL = np.arange(0, 1, 1 / len(GRID_MEAN_LEVEL))
CDF_MEAN_LENGTH = np.arange(0, 1, 1 / len(GRID_MEAN_LENGTH))
TAU_MEAN_LEVEL = 0.01
TAU_MEAN_LENGTH = 0.001

MIN_SPIKE_LEVEL = 0
MIN_SPIKE_P = 0
MAX_SPIKE_LEVEL = 20
MAX_SPIKE_P = 0.05
SPIKE_LEVEL_STEP = 2
SPIKE_P_STEP = 0.01
RANGE_SPIKE_LEVEL = 5
RANGE_SPIKE_P = 0.02
GRID_SPIKE_LEVEL = np.arange(MIN_SPIKE_LEVEL + SPIKE_LEVEL_STEP, MAX_SPIKE_LEVEL + SPIKE_LEVEL_STEP, SPIKE_LEVEL_STEP)
GRID_SPIKE_P = np.round(np.arange(MIN_SPIKE_P + SPIKE_P_STEP, MAX_SPIKE_P + SPIKE_P_STEP, SPIKE_P_STEP), 2)
CDF_SPIKE_LEVEL = np.arange(0, 1, 1 / len(GRID_SPIKE_LEVEL))
CDF_SPIKE_P = np.arange(0, 1, 1 / len(GRID_SPIKE_P))
TAU_SPIKE_LEVEL = 0.2
TAU_SPIKE_P = 0.001

MIN_AMPLITUDE_LEVEL = [0.1, 2]
MIN_AMPLITUDE_LENGTH = 0.2
MAX_AMPLITUDE_LEVEL = [0.9, 10]
MAX_AMPLITUDE_LENGTH = 0.5
AMPLITUDE_LEVEL_STEP = [0.1, 1]
AMPLITUDE_LENGTH_STEP = 0.01
RANGE_AMPLITUDE_LEVEL = [0.2, 2]
RANGE_AMPLITUDE_LENGTH = 0.1
GRID_AMPLITUDE_LEVEL = [np.arange(MIN_AMPLITUDE_LEVEL[i], MAX_AMPLITUDE_LEVEL[i] + AMPLITUDE_LEVEL_STEP[i],
                                  AMPLITUDE_LEVEL_STEP[i]) for i in range(2)]
GRID_AMPLITUDE_LENGTH = np.arange(MIN_AMPLITUDE_LENGTH, MAX_AMPLITUDE_LENGTH + AMPLITUDE_LENGTH_STEP,
                                  AMPLITUDE_LENGTH_STEP)
CDF_AMPLITUDE_LEVEL = [np.arange(0, 1, 1 / len(GRID_AMPLITUDE_LEVEL[i])) for i in range(2)]
CDF_AMPLITUDE_LENGTH = np.arange(0, 1, 1 / len(GRID_AMPLITUDE_LENGTH))
TAU_AMPLITUDE_LEVEL = [0.01, 0.1]
TAU_AMPLITUDE_LENGTH = 0.001

MIN_TREND_SLOPE = -0.01
MIN_TREND_LENGTH = 0.2
MAX_TREND_SLOPE = 0.01
MAX_TREND_LENGTH = 0.5
TREND_SLOPE_STEP = 0.001
TREND_LENGTH_STEP = 0.01
RANGE_TREND_SLOPE = 0.005
RANGE_TREND_LENGTH = 0.1
GRID_TREND_SLOPE = np.arange(MIN_TREND_SLOPE, MAX_TREND_SLOPE + TREND_SLOPE_STEP, TREND_SLOPE_STEP)
GRID_TREND_SLOPE = GRID_TREND_SLOPE[GRID_TREND_SLOPE != 0]
GRID_TREND_LENGTH = np.arange(MIN_TREND_LENGTH, MAX_TREND_LENGTH + TREND_LENGTH_STEP,
                              TREND_LENGTH_STEP)
CDF_TREND_SLOPE = np.arange(0, 1, 1 / len(GRID_TREND_SLOPE))
CDF_TREND_LENGTH = np.arange(0, 1, 1 / len(GRID_TREND_LENGTH))
TAU_TREND_SLOPE = 0.0001
TAU_TREND_LENGTH = 0.001

MIN_VARIANCE_LEVEL = 0.1
MIN_VARIANCE_LENGTH = 0.2
MAX_VARIANCE_LEVEL = 0.5
MAX_VARIANCE_LENGTH = 0.5
VARIANCE_LEVEL_STEP = 0.05
VARIANCE_LENGTH_STEP = 0.01
RANGE_VARIANCE_LEVEL = 0.1
RANGE_VARIANCE_LENGTH = 0.1
GRID_VARIANCE_LEVEL = np.arange(MIN_VARIANCE_LEVEL, MAX_VARIANCE_LEVEL + VARIANCE_LEVEL_STEP, VARIANCE_LEVEL_STEP)
GRID_VARIANCE_LENGTH = np.arange(MIN_VARIANCE_LENGTH, MAX_VARIANCE_LENGTH + VARIANCE_LENGTH_STEP, VARIANCE_LENGTH_STEP)
CDF_VARIANCE_LEVEL = np.arange(0, 1, 1 / len(GRID_VARIANCE_LEVEL))
CDF_VARIANCE_LENGTH = np.arange(0, 1, 1 / len(GRID_VARIANCE_LENGTH))
TAU_VARIANCE_LEVEL = 0.005
TAU_VARIANCE_LENGTH = 0.001

LENGTH_BINS = [0.2, 0.3, 0.4, 0.5]
LEVEL_BINS = [-1, -0.33, 0.33, 1]

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

        # self.anomaly_types = ['platform', 'mean', 'spike', 'amplitude', 'trend', 'variance']
        self.anomaly_types = ['variance']
        # self.anomaly_types = ['platform']

    def forward(self, x):
        x = self.encoder(x)
        return x

    def inject_platform(self, ts_row, level, start, length):
        start_a = int(len(ts_row) * start)
        length_a = int(len(ts_row) * length)
        ts_row[start_a: start_a + length_a] = float(level)
        return ts_row

    def inject_mean(self, ts_row, level, start, length):
        start_a = int(len(ts_row) * start)
        length_a = int(len(ts_row) * length)
        ts_row[start_a: start_a + length_a] += float(level)
        return ts_row

    def inject_spike(self, ts_row, level, p):
        mask = torch.tensor(bernoulli.rvs(p=p, size=len(ts_row)).astype(bool), device=ts_row.device)
        modified_values = ts_row[mask] * level
        modified_values[(ts_row[mask] > 0) & (modified_values < 1)] = 1
        modified_values[(ts_row[mask] < 0) & (modified_values > -1)] = -1
        ts_row[mask] = modified_values
        return ts_row

    def inject_amplitude(self, ts_row, level, start, length):
        start_a = int(len(ts_row) * start)
        length_a = int(len(ts_row) * length)
        amplitude_bell = torch.tensor(np.ones(length_a) * level, device=ts_row.device)
        ts_row[start_a: start_a + length_a] *= amplitude_bell
        return ts_row

    def inject_trend(self, ts_row, slope, start, length):
        start_a = int(len(ts_row) * start)
        length_a = int(len(ts_row) * length)
        slope_a = torch.tensor(np.arange(0, length_a) * slope, device=ts_row.device)
        ts_row[start_a: start_a + length_a] += slope_a
        ts_row[start_a + length_a:] += torch.tensor(np.full(len(ts_row) - start_a - length_a, slope_a[-1].cpu().item()),
                                                    device=ts_row.device)
        return ts_row

    def inject_variance(self, ts_row, level, start, length):
        start_a = int(len(ts_row) * start)
        length_a = int(len(ts_row) * length)
        var = torch.tensor(np.random.normal(0, level, length_a), device=ts_row.device)
        ts_row[start_a: start_a + length_a] += var
        return ts_row

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
                if anomaly_type == 'platform':
                    m = [config_from_grid(CDF_PLATFORM_LEVEL, GRID_PLATFORM_LEVEL), np.random.uniform(0, 0.5),
                         config_from_grid(CDF_PLATFORM_LENGTH, GRID_PLATFORM_LENGTH)]
                    y[i][0] = self.inject_platform(y[i][0], *m)
                    m_pos = [m[0] + np.random.uniform(low=-TAU_PLATFORM_LEVEL, high=TAU_PLATFORM_LEVEL),
                             np.random.uniform(0, 0.5),
                             max(m[2] + np.random.uniform(low=-TAU_PLATFORM_LENGTH, high=TAU_PLATFORM_LENGTH), 0)]
                    y_pos[i][0] = self.inject_platform(y_pos[i][0], *m_pos)
                    for neg_index in range(NUM_NEGATIVE):
                        m_neg = [m[0] + np.random.uniform(low=-RANGE_PLATFORM_LEVEL, high=-TAU_PLATFORM_LEVEL) \
                                     if np.random.random() > 0.5 else
                                 m[0] + np.random.uniform(low=TAU_PLATFORM_LEVEL, high=RANGE_PLATFORM_LEVEL),
                                 np.random.uniform(0, 0.5),
                                 max(m[2] + np.random.uniform(low=-RANGE_PLATFORM_LENGTH, high=-TAU_PLATFORM_LENGTH)
                                     if np.random.random() > 0.5 else
                                     m[2] + np.random.uniform(low=TAU_PLATFORM_LENGTH, high=RANGE_PLATFORM_LENGTH), 0)]
                        y_neg[neg_index][i][0] = self.inject_platform(y_neg[neg_index][i][0], *m_neg)
                        meta_neg[neg_index].append(m_neg)
                elif anomaly_type == 'mean':
                    m = [config_from_grid(CDF_MEAN_LEVEL, GRID_MEAN_LEVEL), np.random.uniform(0, 0.5),
                         config_from_grid(CDF_MEAN_LENGTH, GRID_MEAN_LENGTH)]
                    y[i][0] = self.inject_mean(y[i][0], *m)
                    m_pos = [m[0] + np.random.uniform(low=-TAU_MEAN_LEVEL, high=TAU_MEAN_LEVEL),
                             np.random.uniform(0, 0.5),
                             max(m[2] + np.random.uniform(low=-TAU_MEAN_LENGTH, high=TAU_MEAN_LENGTH), 0)]
                    y_pos[i][0] = self.inject_mean(y_pos[i][0], *m_pos)
                    for neg_index in range(NUM_NEGATIVE):
                        m_neg = [m[0] + np.random.uniform(low=-RANGE_MEAN_LEVEL, high=-TAU_MEAN_LEVEL) \
                                     if np.random.random() > 0.5 else
                                 m[0] + np.random.uniform(low=TAU_MEAN_LEVEL, high=RANGE_MEAN_LEVEL),
                                 np.random.uniform(0, 0.5),
                                 max(m[2] + np.random.uniform(low=-RANGE_MEAN_LENGTH, high=-TAU_MEAN_LENGTH)
                                     if np.random.random() > 0.5 else
                                     m[2] + np.random.uniform(low=TAU_MEAN_LENGTH, high=RANGE_MEAN_LENGTH), 0)]
                        y_neg[neg_index][i][0] = self.inject_mean(y_neg[neg_index][i][0], *m_neg)
                        meta_neg[neg_index].append(m_neg)
                elif anomaly_type == 'spike':
                    m = [config_from_grid(CDF_SPIKE_LEVEL, GRID_SPIKE_LEVEL),
                         config_from_grid(CDF_SPIKE_P, GRID_SPIKE_P)]
                    y[i][0] = self.inject_spike(y[i][0], *m)
                    m_pos = [m[0] + np.random.uniform(low=-TAU_SPIKE_LEVEL, high=TAU_SPIKE_LEVEL),
                             max(m[1] + np.random.uniform(low=-TAU_SPIKE_P, high=TAU_SPIKE_P), 0)]
                    y_pos[i][0] = self.inject_spike(y_pos[i][0], *m_pos)
                    for neg_index in range(NUM_NEGATIVE):
                        m_neg = [m[0] + np.random.uniform(low=-RANGE_SPIKE_LEVEL, high=-TAU_SPIKE_LEVEL) \
                                     if np.random.random() > 0.5 else
                                 m[0] + np.random.uniform(low=TAU_SPIKE_LEVEL, high=RANGE_SPIKE_LEVEL),
                                 max(m[1] + np.random.uniform(low=-RANGE_SPIKE_P, high=-TAU_SPIKE_P)
                                     if np.random.random() > 0.5 else
                                     m[1] + np.random.uniform(low=TAU_SPIKE_P, high=RANGE_SPIKE_P), 0)]
                        y_neg[neg_index][i][0] = self.inject_spike(y_neg[neg_index][i][0], *m_neg)
                        meta_neg[neg_index].append(m_neg)

                    # m_pos = [np.random.uniform(low=m[0] - TAU_SPIKE_LEVEL,
                    #                            high=min(m[0] + TAU_SPIKE_LEVEL, MAX_SPIKE_LEVEL)),
                    #          np.random.uniform(low=m[1] - TAU_SPIKE_P, high=min(m[1] + TAU_SPIKE_P, MAX_SPIKE_P))]
                    # y_pos[i][0] = self.inject_spike(y_pos[i][0], *m_pos)
                    # for neg_index in range(NUM_NEGATIVE):
                    #     m_neg = [np.random.uniform(low=MIN_SPIKE_LEVEL, high=m[0] - TAU_SPIKE_LEVEL)
                    #              if np.random.random() < ((m[0] - MIN_SPIKE_LEVEL - TAU_SPIKE_LEVEL)
                    #                                       / MAX_SPIKE_LEVEL - MIN_SPIKE_LEVEL - 2 * TAU_SPIKE_LEVEL)
                    #              else np.random.uniform(low=m[0] + TAU_SPIKE_LEVEL, high=MAX_SPIKE_LEVEL),
                    #              np.random.uniform(low=MIN_SPIKE_P, high=m[1] - TAU_SPIKE_P)
                    #              if np.random.random() < ((m[1] - MIN_SPIKE_P - TAU_SPIKE_P)
                    #                                       / MAX_SPIKE_P - MIN_SPIKE_P - 2 * TAU_SPIKE_P)
                    #              else np.random.uniform(low=m[1] + TAU_SPIKE_P, high=MAX_SPIKE_P)]
                    #     y_neg[neg_index][i][0] = self.inject_spike(y_neg[neg_index][i][0], *m_neg)
                    #     meta_neg[neg_index].append(m_neg)
                elif anomaly_type == 'amplitude':
                    index = 0
                    if np.random.random() > 0.5:
                        index = 1
                    m = [config_from_grid(CDF_AMPLITUDE_LEVEL[index], GRID_AMPLITUDE_LEVEL[index]),
                         np.random.uniform(0, 0.5), config_from_grid(CDF_AMPLITUDE_LENGTH, GRID_AMPLITUDE_LENGTH)]
                    y[i][0] = self.inject_amplitude(y[i][0], *m)
                    m_pos = [np.random.uniform(low=m[0] - TAU_AMPLITUDE_LEVEL[index],
                                               high=min(m[0] + TAU_AMPLITUDE_LEVEL[index], MAX_AMPLITUDE_LEVEL[index])),
                             np.random.uniform(0, 0.5),
                             np.random.uniform(low=m[2] - TAU_AMPLITUDE_LENGTH,
                                               high=min(m[2] + TAU_AMPLITUDE_LENGTH, MAX_AMPLITUDE_LENGTH))]
                    y_pos[i][0] = self.inject_amplitude(y_pos[i][0], *m_pos)
                    for neg_index in range(NUM_NEGATIVE):
                        m_neg = [
                            np.random.uniform(low=MIN_AMPLITUDE_LEVEL[index], high=m[0] - TAU_AMPLITUDE_LEVEL[index])
                            if np.random.random() < ((m[0] - MIN_AMPLITUDE_LEVEL[index] - TAU_AMPLITUDE_LEVEL[index])
                                                     / MAX_AMPLITUDE_LEVEL[index] - MIN_AMPLITUDE_LEVEL[index] - 2 *
                                                     TAU_AMPLITUDE_LEVEL[index])
                            else np.random.uniform(low=m[0] + TAU_AMPLITUDE_LEVEL[index],
                                                   high=MAX_AMPLITUDE_LEVEL[index]),
                            np.random.uniform(0, 0.5),
                            np.random.uniform(low=MIN_AMPLITUDE_LENGTH, high=m[2] - TAU_AMPLITUDE_LENGTH)
                            if np.random.random() < ((m[2] - MIN_AMPLITUDE_LENGTH - TAU_AMPLITUDE_LENGTH)
                                                     / MAX_AMPLITUDE_LENGTH - MIN_AMPLITUDE_LENGTH - 2 *
                                                     TAU_AMPLITUDE_LENGTH)
                            else np.random.uniform(low=m[2] + TAU_AMPLITUDE_LENGTH, high=MAX_AMPLITUDE_LENGTH)]
                        y_neg[neg_index][i][0] = self.inject_amplitude(y_neg[neg_index][i][0], *m_neg)
                        meta_neg[neg_index].append(m_neg)
                elif anomaly_type == 'trend':
                    m = [config_from_grid(CDF_TREND_SLOPE, GRID_TREND_SLOPE), np.random.uniform(0, 0.5),
                         config_from_grid(CDF_TREND_LENGTH, GRID_TREND_LENGTH)]
                    y[i][0] = self.inject_trend(y[i][0], *m)
                    m_pos = [np.random.uniform(low=m[0] - TAU_TREND_SLOPE,
                                               high=min(m[0] + TAU_TREND_SLOPE, MAX_TREND_SLOPE)),
                             np.random.uniform(0, 0.5),
                             np.random.uniform(low=m[2] - TAU_TREND_LENGTH,
                                               high=min(m[2] + TAU_TREND_LENGTH, MAX_TREND_LENGTH))]
                    y_pos[i][0] = self.inject_trend(y_pos[i][0], *m_pos)
                    for neg_index in range(NUM_NEGATIVE):
                        m_neg = [np.random.uniform(low=MIN_TREND_SLOPE, high=m[0] - TAU_TREND_SLOPE)
                                 if np.random.random() < ((m[0] - MIN_TREND_SLOPE - TAU_TREND_SLOPE)
                                                          / MAX_TREND_SLOPE - MIN_TREND_SLOPE - 2 * TAU_TREND_SLOPE)
                                 else np.random.uniform(low=m[0] + TAU_TREND_SLOPE, high=MAX_TREND_SLOPE),
                                 np.random.uniform(0, 0.5),
                                 np.random.uniform(low=MIN_TREND_LENGTH, high=m[2] - TAU_TREND_LENGTH)
                                 if np.random.random() < ((m[2] - MIN_TREND_LENGTH - TAU_TREND_LENGTH)
                                                          / MAX_TREND_LENGTH - MIN_TREND_LENGTH - 2 * TAU_TREND_LENGTH)
                                 else np.random.uniform(low=m[2] + TAU_TREND_LENGTH, high=MAX_TREND_LENGTH)]
                        y_neg[neg_index][i][0] = self.inject_trend(y_neg[neg_index][i][0], *m_neg)
                        meta_neg[neg_index].append(m_neg)
                elif anomaly_type == 'variance':
                    m = [config_from_grid(CDF_VARIANCE_LEVEL, GRID_VARIANCE_LEVEL), np.random.uniform(0, 0.5),
                         config_from_grid(CDF_VARIANCE_LENGTH, GRID_VARIANCE_LENGTH)]
                    y[i][0] = self.inject_variance(y[i][0], *m)
                    m_pos = [np.random.uniform(low=m[0] - TAU_VARIANCE_LEVEL,
                                               high=min(m[0] + TAU_VARIANCE_LEVEL, MAX_VARIANCE_LEVEL)),
                             np.random.uniform(0, 0.5),
                             np.random.uniform(low=m[2] - TAU_VARIANCE_LENGTH,
                                               high=min(m[2] + TAU_VARIANCE_LENGTH, MAX_VARIANCE_LENGTH))]
                    y_pos[i][0] = self.inject_variance(y_pos[i][0], *m_pos)
                    for neg_index in range(NUM_NEGATIVE):
                        m_neg = [np.random.uniform(low=MIN_VARIANCE_LEVEL, high=m[0] - TAU_VARIANCE_LEVEL)
                                 if np.random.random() < ((m[0] - MIN_VARIANCE_LEVEL - TAU_VARIANCE_LEVEL)
                                                          / MAX_VARIANCE_LEVEL - MIN_VARIANCE_LEVEL - 2 *
                                                          TAU_VARIANCE_LEVEL)
                                 else np.random.uniform(low=m[0] + TAU_VARIANCE_LEVEL, high=MAX_VARIANCE_LEVEL),
                                 np.random.uniform(0, 0.5),
                                 np.random.uniform(low=MIN_VARIANCE_LENGTH, high=m[2] - TAU_VARIANCE_LENGTH)
                                 if np.random.random() < ((m[2] - MIN_VARIANCE_LENGTH - TAU_VARIANCE_LENGTH)
                                                          / MAX_VARIANCE_LENGTH - MIN_VARIANCE_LENGTH - 2 *
                                                          TAU_VARIANCE_LENGTH)
                                 else np.random.uniform(low=m[2] + TAU_VARIANCE_LENGTH, high=MAX_VARIANCE_LENGTH)]
                        y_neg[neg_index][i][0] = self.inject_variance(y_neg[neg_index][i][0], *m_neg)
                        meta_neg[neg_index].append(m_neg)
                else:
                    raise Exception(f'Unsupported anomaly_type: {anomaly_type}')
                meta.append(m)
                meta_pos.append(m_pos)

            c_y_dict[anomaly_type] = self(y)
            c_y_pos_dict[anomaly_type] = self(y_pos)
            c_y_neg_dict[anomaly_type] = [self(y_neg[i]) for i in range(NUM_NEGATIVE)]
            meta_dict[anomaly_type] = meta
            meta_neg_dict[anomaly_type] = meta_neg

        ### Anomalies should be close to the ones with the same type and similar hyperparameters, and far away from
        # the ones with different types and normal.
        loss_global = 0
        for anomaly_type in self.anomaly_types:
            _c_y = list()
            for _anomaly_type in self.anomaly_types:
                if _anomaly_type != anomaly_type:
                    _c_y.append(c_y_dict[_anomaly_type])
            if len(_c_y) == 0:
                loss_global += self.info_loss(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type],
                                              torch.cat([c_x, c_x_pos], dim=0))
            else:
                _c_y = torch.cat(_c_y, dim=0)
                loss_global += self.info_loss(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type],
                                              torch.cat([c_x, c_x_pos, _c_y], dim=0))
        loss_global /= len(self.anomaly_types)

        ### Anomalies with far away hyperparameters should be far away propotional to delta.
        loss_local = 0
        for anomaly_type in self.anomaly_types:
            loss_local += sum([hard_negative_loss(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type],
                                                  c_y_neg_dict[anomaly_type][i], np.array(meta_dict[anomaly_type]),
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
            raise Exception(f'loss_global: {loss_global}, loss_local: {loss_local}, loss_normal: {loss_normal}')

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

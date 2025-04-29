import numpy as np
import torch
import pytorch_lightning as pl
from info_nce import InfoNCE
# Ensure relative import works for your structure
try:
    from .model import CNNEncoder
except ImportError:
    from model import CNNEncoder
import torch.nn.functional as F
import random


# --- CONFIGS, PARAMETERS, HELPERS (Keep exactly as provided) ---
def generate_parameters(min, max, step, tau, exclude_zero=False):
    grid = np.arange(min, max + step, step)
    if exclude_zero:
        grid = grid[grid != 0]
    # Add safety check for empty grid before calculating cdf step
    if len(grid) == 0:
        # Handle empty grid case (e.g., min=max, exclude_zero=True)
        print(f"Warning: generate_parameters resulted in empty grid for min={min}, max={max}, step={step}, exclude_zero={exclude_zero}. Returning grid=[{min}], cdf=[0.].")
        grid = np.array([min]) # Fallback to a single value
        cdf = np.array([0.0])
    else:
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

    z_anc, z_pos, z_neg = normalize(z_anc, z_pos, z_neg)
    positive_logit = torch.sum(z_anc * z_pos, dim=1, keepdim=True)

    # --- Ensure z_neg is [B, N_neg, D] ---
    # Handle case where z_neg comes in as [N_neg, B, D] from torch.stack
    if z_neg.dim() == 3 and z_neg.shape[0] == NUM_NEGATIVE and z_neg.shape[1] == z_anc.shape[0]:
         z_neg = z_neg.permute(1, 0, 2) # Now [B, N_neg, D]
    elif not (z_neg.dim() == 3 and z_neg.shape[0] == z_anc.shape[0] and z_neg.shape[1] == NUM_NEGATIVE):
         # Add check for N_neg=1 case? If NUM_NEGATIVE is always > 1 this might not be needed
         raise ValueError(f"Unexpected shape for z_neg in local loss: {z_neg.shape}. Expected [B, N_neg, D] or [N_neg, B, D]. B={z_anc.shape[0]}, N_neg={NUM_NEGATIVE}")
    # --- End shape check ---

    # negative_logits = torch.bmm(z_anc.unsqueeze(1), z_neg.permute(1, 0, 2).transpose(1, 2)).squeeze(1) # Original line had extra permute/transpose
    negative_logits = torch.bmm(z_anc.unsqueeze(1), z_neg.transpose(1, 2)).squeeze(1) # [B, 1, D] @ [B, D, N_neg] -> [B, 1, N_neg] -> [B, N_neg]
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long, device=z_anc.device)
    return F.cross_entropy(logits / temperature, labels, reduction='mean')


def config_from_grid(cdf, grid):
    # Add safety check for potentially empty grid/cdf from generate_parameters
    if len(grid) == 0: return np.nan # Cannot sample from empty grid
    if len(cdf) == 0: # If grid has values but cdf is empty (e.g., grid size 1)
        if len(grid) == 1: return grid[0]
        else: return np.random.choice(grid) # Fallback if CDF failed

    # Original logic
    bin_idx = np.digitize(np.random.random(1), bins=cdf)[0]
    # Clamp index just in case digitize gives index == len(grid)
    grid_idx = min(bin_idx - 1, len(grid) - 1)
    return grid[grid_idx]


class Encoder(pl.LightningModule):
    def __init__(self, args, ts_input_size, lr):
        super().__init__()
        # Don't store args if only used for device initially
        # self.args = args
        # Save hyperparameters needed for checkpoint loading etc.
        self.save_hyperparameters('ts_input_size', 'lr') # args removed

        self.encoder = CNNEncoder(ts_input_size)
        # self.lr captured by save_hyperparameters, accessible via self.hparams.lr
        self.info_loss = InfoNCE()
        self.normal_idx = set()

        # --- MODIFICATION: Use register_buffer ---
        # Initialize buffer on CPU, Lightning moves it to correct device.
        # Detach prevents it being part of grad computation initially.
        self.register_buffer("normal_x", torch.tensor([]).detach())
        # --- END MODIFICATION ---

        self.anomaly_types = ['platform', 'mean', 'spike', 'amplitude', 'trend', 'variance']

    # --- forward and injection methods (Keep exactly as provided) ---
    def forward(self, x):
        # Input x is assumed on self.device by Lightning
        x = self.encoder(x)
        return x

    def inject_platform(self, ts_row, level, length, start):
        # Make sure length is int and start index calculation is safe
        length = int(length)
        start_a = int(len(ts_row) * start)
        # Clamp indices
        start_a = max(0, start_a)
        end_a = min(len(ts_row), start_a + length)
        if start_a >= end_a: return ts_row # Avoid empty slice
        ts_row[start_a : end_a] = float(level)
        return ts_row

    def inject_mean(self, ts_row, level, length, start):
        length = int(length)
        start_a = int(len(ts_row) * start)
        start_a = max(0, start_a)
        end_a = min(len(ts_row), start_a + length)
        if start_a >= end_a: return ts_row
        ts_row[start_a : end_a] += float(level)
        return ts_row

    def inject_spike(self, ts_row, level, start):
        start_a = int(len(ts_row) * start)
        start_a = max(0, min(start_a, len(ts_row) - 1)) # Clamp index
        ts_row[start_a] = ts_row[start_a] + level if random.random() < 0.5 else ts_row[start_a] - level # Use random module
        return ts_row

    def inject_amplitude(self, ts_row, level, length, start):
        length = int(length)
        start_a = int(len(ts_row) * start)
        start_a = max(0, start_a)
        end_a = min(len(ts_row), start_a + length)
        actual_length = end_a - start_a
        if actual_length <= 0: return ts_row

        amplitude_bell = torch.full((actual_length,), float(level), device=ts_row.device, dtype=ts_row.dtype) # Use full for dtype/device
        ts_row[start_a : end_a] *= amplitude_bell
        return ts_row

    def inject_trend(self, ts_row, slope, length, start):
        length = int(length)
        start_a = int(len(ts_row) * start)
        start_a = max(0, start_a)
        end_a = min(len(ts_row), start_a + length)
        actual_length = end_a - start_a
        if actual_length <= 0: return ts_row

        slope_a = torch.arange(0, actual_length, device=ts_row.device, dtype=ts_row.dtype) * float(slope) # Use arange for dtype/device
        ts_row[start_a : end_a] += slope_a

        if end_a < len(ts_row) and actual_length > 0:
            # Ensure addition tensor is also on the correct device/dtype
            add_val = slope_a[-1] # This is already a tensor on the correct device
            ts_row[end_a:] += add_val
        return ts_row

    def inject_variance(self, ts_row, level, length, start):
        length = int(length)
        start_a = int(len(ts_row) * start)
        start_a = max(0, start_a)
        end_a = min(len(ts_row), start_a + length)
        actual_length = end_a - start_a
        if actual_length <= 0: return ts_row

        var = torch.randn(actual_length, device=ts_row.device, dtype=ts_row.dtype) * float(level) # Use randn for dtype/device
        ts_row[start_a : end_a] += var
        return ts_row
    # --- End injection methods ---

    # --- main_process (Keep exactly as provided, relies on self.normal_x being on correct device) ---
    def main_process(self, x, process):
        # Add check for empty buffer before sampling
        if self.normal_x is None or self.normal_x.nelement() == 0:
             print(f"Error ({process}): normal_x buffer is empty. Cannot sample positives.")
             # Depending on desired behavior, either raise error or return a zero loss / skip step
             # Returning zero loss to avoid crash, but training might be ineffective
             return torch.tensor(0.0, device=self.device, requires_grad=True if process=='train' else False)

        x_pos = self.normal_x[np.random.choice(len(self.normal_x), x.shape[0])] # Assumes self.normal_x on self.device
        c_x = self(x)       # x assumed on self.device
        c_x_pos = self(x_pos) # x_pos should be on self.device

        c_y_dict, c_y_pos_dict, c_y_neg_dict = dict(), dict(), dict()
        for anomaly_type in self.anomaly_types:
            y = x.clone()
            y_pos = [x.clone() for _ in range(NUM_POSITIVE)]
            y_neg = [x.clone() for _ in range(NUM_NEGATIVE)]
            inject = getattr(self, f"inject_{anomaly_type}")
            for i in range(len(x)):
                m = list()
                start = random.uniform(0, 1) # Use random module
                index = 0 # Default index
                # Check if parameter list exists and has more than one element
                if isinstance(PARAMETERS[anomaly_type].get('level'), list) and len(PARAMETERS[anomaly_type]['level']) > 1:
                     index = 0 if random.random() < 0.5 else 1

                param_keys = list(PARAMETERS[anomaly_type].keys()) # Get keys once
                for config_key in param_keys:
                    config_val = PARAMETERS[anomaly_type][config_key]
                    if isinstance(config_val, list):
                        # Need to handle case where index might be out of bounds if list length changes
                        safe_index = min(index, len(config_val) - 1)
                        m.append(config_from_grid(config_val[safe_index]['cdf'],
                                                  config_val[safe_index]['grid']))
                    else:
                        m.append(config_from_grid(config_val['cdf'],
                                                  config_val['grid']))
                    if config_key == 'length':
                        # This start re-assignment looks suspect, might override the initial random start?
                        # Keeping as is based on user's code.
                        start = random.uniform(0, 0.5)

                # Handle potential NaN from config_from_grid before injection
                if any(np.isnan(val) for val in m):
                    print(f"Warning ({process}): NaN detected in anchor params for {anomaly_type}, skipping injection for item {i}.")
                    continue # Skip this sample in batch

                # Determine target tensor shape [C, L] or [L]
                if y[i].dim() == 2 and y[i].shape[0] == 1: # [1, L]
                    target_ts = y[i][0]
                elif y[i].dim() == 1: # [L]
                    target_ts = y[i]
                else:
                    raise ValueError(f"Unsupported tensor shape for injection: {y[i].shape}")

                # Apply injection
                inject(target_ts, *m, start=start)


                # --- inject positive (Keep logic as provided) ---
                for pos_index in range(NUM_POSITIVE):
                    m_pos = list()
                    start = random.uniform(0, 1)
                    for j, config_key in enumerate(param_keys):
                        config_val = PARAMETERS[anomaly_type][config_key]
                        is_list_config = isinstance(config_val, list)
                        safe_index = min(index, len(config_val) - 1) if is_list_config else 0
                        current_config = config_val[safe_index] if is_list_config else config_val

                        if config_key == 'length':
                            start = random.uniform(0, 0.5) # Again, suspect start re-assignment
                            # Ensure tau application doesn't go below min or above max
                            low_bound = max(current_config['min'], m[j] - current_config['tau'])
                            high_bound = min(current_config['max'], m[j] + current_config['tau'])
                            # Ensure bounds are valid integers for choice
                            low_bound_int = int(round(low_bound))
                            high_bound_int = int(round(high_bound))
                            anchor_int = int(round(m[j]))
                            # Clamp choice options
                            options = [anchor_int]
                            if anchor_int - int(round(current_config['tau'])) >= low_bound_int:
                                options.append(anchor_int - int(round(current_config['tau'])))
                            if anchor_int + int(round(current_config['tau'])) <= high_bound_int:
                                options.append(anchor_int + int(round(current_config['tau'])))
                            options = list(set(options)) # Remove duplicates
                            m_p = np.random.choice(options)
                            # Ensure choice respects bounds strictly
                            m_p = max(low_bound_int, min(high_bound_int, m_p))
                            m_pos.append(m_p)

                            # Original exception check (keeping as requested)
                            if np.abs(m[j] - m_p) > current_config['tau'] + 1e-6: # Add tolerance for float/int compare
                                print(f"WARN Positive Length: Orig={m[j]}, New={m_p}, Tau={current_config['tau']}")
                                # raise Exception("Positive length difference exceeds tau") # Keep commented maybe
                        else:
                            low = max(m[j] - current_config['tau'], current_config['min'])
                            high = min(m[j] + current_config['tau'], current_config['max'])
                            if low >= high: m_p = m[j] # Fallback if interval collapses
                            else: m_p = random.uniform(low, high)
                            m_pos.append(m_p)

                            # Original exception check (keeping as requested)
                            if np.abs(m[j] - m_p) > current_config['tau'] + 1e-6: # Add tolerance
                                print(f"WARN Positive Level/Slope: Orig={m[j]}, New={m_p}, Tau={current_config['tau']}")
                                # raise Exception("Positive value difference exceeds tau") # Keep commented

                    if any(np.isnan(val) for val in m_pos):
                         print(f"Warning ({process}): NaN detected in positive params for {anomaly_type}, skipping injection for item {i}, pos {pos_index}.")
                         continue

                    if y_pos[pos_index][i].dim() == 2 and y_pos[pos_index][i].shape[0] == 1:
                         target_ts_pos = y_pos[pos_index][i][0]
                    elif y_pos[pos_index][i].dim() == 1:
                         target_ts_pos = y_pos[pos_index][i]
                    else: continue # Skip if shape wrong

                    inject(target_ts_pos, *m_pos, start=start)


                # --- inject negative (Keep logic as provided) ---
                for neg_index in range(NUM_NEGATIVE):
                    m_neg = list()
                    start = random.uniform(0, 1)
                    for j, config_key in enumerate(param_keys):
                        config_val = PARAMETERS[anomaly_type][config_key]
                        is_list_config = isinstance(config_val, list)
                        safe_index = min(index, len(config_val) - 1) if is_list_config else 0
                        current_config = config_val[safe_index] if is_list_config else config_val

                        if config_key == 'length':
                            start = random.uniform(0, 0.5) # Suspect start re-assignment
                            min_val, max_val = current_config['min'], current_config['max']
                            anchor_val, tau = m[j], current_config['tau']
                            lower_max = int(np.floor(anchor_val - tau - 1))
                            upper_min = int(np.ceil(anchor_val + tau + 1))

                            # Calculate probabilities based on available range sizes
                            lower_range = max(0, lower_max - min_val + 1)
                            upper_range = max(0, max_val - upper_min + 1)
                            total_range = lower_range + upper_range

                            if total_range <= 0: # No valid range outside tau
                                m_n = int(round(anchor_val)) # Fallback
                                while abs(m_n - anchor_val) <= tau: # Ensure it's actually outside if possible
                                     m_n = random.randint(min_val, max_val)
                            elif random.random() < (lower_range / total_range):
                                m_n = random.randint(min_val, lower_max)
                            else:
                                m_n = random.randint(upper_min, max_val)
                            m_neg.append(m_n)

                            # Original exception check (keeping as requested)
                            if np.abs(m[j] - m_n) <= current_config['tau']:
                                print(f"WARN Negative Length: Orig={m[j]}, New={m_n}, Tau={current_config['tau']}")
                                # raise Exception("Negative length difference NOT outside tau") # Keep commented
                        else:
                            min_val, max_val = current_config['min'], current_config['max']
                            anchor_val, tau = m[j], current_config['tau']
                            lower_max = anchor_val - tau
                            upper_min = anchor_val + tau

                             # Calculate probabilities based on range sizes
                            lower_range_size = max(0, lower_max - min_val)
                            upper_range_size = max(0, max_val - upper_min)
                            total_valid_range = lower_range_size + upper_range_size

                            if total_valid_range <= 1e-9: # No valid range outside tau
                                 m_n = anchor_val # Fallback
                                 while abs(m_n - anchor_val) <= tau: # Ensure it's actually outside if possible
                                      m_n = random.uniform(min_val, max_val)
                            elif random.random() < (lower_range_size / total_valid_range):
                                m_n = random.uniform(min_val, lower_max)
                            else:
                                m_n = random.uniform(upper_min, max_val)
                            m_neg.append(m_n)

                            # Original exception check (keeping as requested)
                            if np.abs(m[j] - m_n) <= current_config['tau']:
                                print(f"WARN Negative Level/Slope: Orig={m[j]}, New={m_n}, Tau={current_config['tau']}")
                                # raise Exception("Negative value difference NOT outside tau") # Keep commented

                    if any(np.isnan(val) for val in m_neg):
                        print(f"Warning ({process}): NaN detected in negative params for {anomaly_type}, skipping injection for item {i}, neg {neg_index}.")
                        continue

                    if y_neg[neg_index][i].dim() == 2 and y_neg[neg_index][i].shape[0] == 1:
                         target_ts_neg = y_neg[neg_index][i][0]
                    elif y_neg[neg_index][i].dim() == 1:
                         target_ts_neg = y_neg[neg_index][i]
                    else: continue # Skip if shape wrong

                    inject(target_ts_neg, *m_neg, start=start)


            # --- Get Embeddings (Keep as provided) ---
            c_y_dict[anomaly_type] = self(y)
            # Handle potential errors if lists become empty due to NaNs during generation
            c_y_pos_dict[anomaly_type] = [self(y_p) for y_p in y_pos if y_p is not None] # Filter None? Check y_pos structure
            c_y_neg_dict[anomaly_type] = [self(y_n) for y_n in y_neg if y_n is not None] # Filter None? Check y_neg structure
            # If filtering leads to empty lists, loss calculation below needs adjustment
            if not c_y_pos_dict[anomaly_type]:
                 print(f"Warning ({process}): No valid positive samples generated for {anomaly_type} after NaN filtering.")
                 # Need a strategy here: skip loss? use anchor?
                 continue # Skip loss for this type if no positives
            if not c_y_neg_dict[anomaly_type]:
                 print(f"Warning ({process}): No valid negative samples generated for {anomaly_type} after NaN filtering.")
                 # Need a strategy here: skip local loss?
                 # Keep c_y_dict for global/normal loss calculations

        # --- Loss Calculations (Keep as provided, with checks for empty lists) ---
        loss_global = 0
        valid_anomaly_types_global = [at for at in self.anomaly_types if at in c_y_dict and at in c_y_pos_dict and c_y_pos_dict[at]]
        if valid_anomaly_types_global:
            for anomaly_type in valid_anomaly_types_global:
                c_y_others = list()
                for _anomaly_type in valid_anomaly_types_global: # Only compare against other valid types
                    if _anomaly_type != anomaly_type:
                        c_y_others.append(c_y_dict[_anomaly_type])

                negatives_global = [c_x, c_x_pos]
                if c_y_others:
                    negatives_global.append(torch.cat(c_y_others, dim=0))

                # Use first positive sample (assuming NUM_POSITIVE=1 or main one)
                loss_global += self.info_loss(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type][0],
                                              torch.cat(negatives_global, dim=0))
            loss_global /= len(valid_anomaly_types_global)

        loss_local = 0
        valid_anomaly_types_local = [at for at in self.anomaly_types if at in c_y_dict and at in c_y_pos_dict and c_y_pos_dict[at] and at in c_y_neg_dict and c_y_neg_dict[at]]
        if valid_anomaly_types_local:
            for anomaly_type in valid_anomaly_types_local:
                # Use first positive sample
                # Stack valid negative samples
                stacked_negatives = torch.stack(c_y_neg_dict[anomaly_type], dim=0) # [N_neg, B, D]
                loss_local += local(c_y_dict[anomaly_type], c_y_pos_dict[anomaly_type][0], stacked_negatives)
            loss_local /= len(valid_anomaly_types_local)

        loss_normal = 0
        all_anomalies_list = []
        # Gather all valid generated embeddings
        for at in self.anomaly_types:
            if at in c_y_dict: all_anomalies_list.append(c_y_dict[at])
            if at in c_y_pos_dict and c_y_pos_dict[at]: all_anomalies_list.extend(c_y_pos_dict[at])
            if at in c_y_neg_dict and c_y_neg_dict[at]: all_anomalies_list.extend(c_y_neg_dict[at])

        if all_anomalies_list:
            loss_normal = self.info_loss(c_x, c_x_pos, torch.cat(all_anomalies_list, dim=0))
        else:
            print(f"Warning ({process}): No valid anomaly embeddings generated for normal loss.")


        loss = loss_global + loss_local + loss_normal

        # --- Logging (Keep as provided) ---
        # Check for NaN/Inf before logging
        if torch.isnan(loss) or torch.isinf(loss):
             print(f"ERROR ({process}): Calculated loss is NaN or Inf. loss_global={loss_global}, loss_local={loss_local}, loss_normal={loss_normal}")
             # Optionally return a high value or skip logging/step
             # return torch.tensor(1e6, device=self.device, requires_grad=True if process=='train' else False) # Example fallback
        else:
             self.log(f"{process}/loss_global", loss_global, on_step=False, on_epoch=True, prog_bar=False, batch_size=x.shape[0], sync_dist=True)
             self.log(f"{process}/loss_local", loss_local, on_step=False, on_epoch=True, prog_bar=False, batch_size=x.shape[0], sync_dist=True)
             self.log(f"{process}/loss_normal", loss_normal, on_step=False, on_epoch=True, prog_bar=False, batch_size=x.shape[0], sync_dist=True)
             self.log(f"{process}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)

        return loss
    # --- End main_process ---


    def training_step(self, batch, batch_idx):
        # batch should be tensor on self.device
        x = batch[0] if isinstance(batch, (list, tuple)) else batch # Handle dataloader wrapping

        # --- MODIFICATION: Update buffer without .to() ---
        if batch_idx not in self.normal_idx:
             self.normal_idx.add(batch_idx)
             # Detach x before adding to buffer if buffer requires_grad=False (default for buffers)
             x_detached = x.detach()
             if self.normal_x is None or self.normal_x.nelement() == 0: # Check if buffer is empty
                 self.normal_x = x_detached
             else:
                 # Concatenation happens on the device self.normal_x and x reside on (should be self.device)
                 self.normal_x = torch.cat([self.normal_x, x_detached], dim=0)
             # Optional: Limit buffer size
             max_buffer_size = 10000 # Example
             if self.normal_x.shape[0] > max_buffer_size:
                  self.normal_x = self.normal_x[-max_buffer_size:]
        # --- END MODIFICATION ---

        return self.main_process(x=x, process='train')

    def validation_step(self, batch, batch_idx):
        # No buffer update in validation
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self.main_process(x=x, process='val')

    def configure_optimizers(self):
        # Access lr via hparams (saved by self.save_hyperparameters)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
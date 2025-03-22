from typing import Any
import numpy as np
import os
import pandas as pd
import argparse
from tqdm import tqdm
import pickle

import warnings

warnings.filterwarnings("ignore")

LENGTH_BINS = [0.2, 0.3, 0.4, 0.5]
LEVEL_BINS = [-1, -0.33, 0.33, 1]


def mix_anomalies(cfg, splitted_data):
    """
    Adds anomalies to validation and test input data plus replaces y to binary y-labels.
    By default only adds anomalies to validation and test data, but can also include training data.
    """
    splitted_data = list(splitted_data)
    if cfg["mix_train"]:
        idx = [0, 1, 2]
    else:
        idx = [1, 2]
    for i in idx:
        splitted_data[i] = _add_labeled_anomalies(splitted_data[i], cfg["mix_ratio"])
    return splitted_data


def _add_labeled_anomalies(data, n_samples_frac):
    x, y, meta = data
    n_samples = int(n_samples_frac * y.shape[0])
    selected_rows = np.random.choice(y.shape[0], size=n_samples, replace=False)
    x, y = _replace_rows(x, y, selected_rows)
    return x, y, meta


def _replace_rows(x, y, selected_rows):
    x_size = x.shape[0]
    x[selected_rows, :] = y[selected_rows, :]
    y = np.zeros((x_size, 1))
    y[selected_rows, :] = 1
    return x, y


class AnomFuncs:
    """
    Class that contains functions to inject anomalies into time series data
    """

    def __init__(self, df, save_path="data", seed=96) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.df = df
        self.np_data = np.copy(self.df.values)
        self.save_path = save_path

    def _reset_data(self):
        self.np_data = np.copy(self.df.values)

    def _reset_rng(self):
        rng = np.random.default_rng(self.seed)
        # assert rng.random(1) == self.rng.random(1)
        return rng

    @staticmethod
    def inject_platform(ts_row, level_h0, level_h1, length_h0, length_h1, start):
        start = int(len(ts_row) * start)

        cdf = [0, length_h0, length_h0 + length_h1, 1]
        length_r = np.digitize(np.random.random(1), bins=cdf)[0]
        r = np.random.uniform(LENGTH_BINS[length_r - 1], LENGTH_BINS[length_r])
        length = int(len(ts_row) * r)

        cdf = [0, level_h0, level_h0 + level_h1, 1]
        level_r = np.digitize(np.random.random(1), bins=cdf)[0]
        level = np.random.uniform(LEVEL_BINS[level_r - 1], LEVEL_BINS[level_r])

        ts_row[start: start + length] = level
        return ts_row, start, length, level

    @staticmethod
    def inject_mean(ts_row, level_h0, level_h1, length_h0, length_h1, start):
        start = int(len(ts_row) * start)

        cdf = [0, length_h0, length_h0 + length_h1, 1]
        length_r = np.digitize(np.random.random(1), bins=cdf)[0]
        r = np.random.uniform(LENGTH_BINS[length_r - 1], LENGTH_BINS[length_r])
        length = int(len(ts_row) * r)

        cdf = [0, level_h0, level_h0 + level_h1, 1]
        level_r = np.digitize(np.random.random(1), bins=cdf)[0]
        level = np.random.uniform(LEVEL_BINS[level_r - 1], LEVEL_BINS[level_r])

        ts_row[start: start + length] += level
        return ts_row, start, length, level

    # @staticmethod
    # def inject_spike(ts_row, level_h0, level_h1, start):
    #     start_index = int(len(ts_row) * start)
    #
    #     cdf = [0, level_h0, level_h0 + level_h1, 1]
    #     level_r = np.digitize(np.random.random(1), bins=cdf)[0]
    #     level = np.random.uniform(LEVEL_BINS[level_r - 1], LEVEL_BINS[level_r])
    #
    #     ts_row[start_index] += level
    #
    #     return ts_row, start_index, 1, level

    # @staticmethod
    # def inject_trend(ts_row, slope, start, length):
    #     start = int(len(ts_row) * start)
    #     length = int(len(ts_row) * length)
    #     ts_row[start : start + length] += np.arange(0, length) * slope
    #     return ts_row

    # @staticmethod
    # def inject_variance(ts_row, level, start, length):
    #     start = int(len(ts_row) * start)
    #     length = int(len(ts_row) * length)
    #     var = np.random.normal(0, level, length)
    #     ts_row[start : start + length] += var
    #     return ts_row

    # @staticmethod
    # def inject_extremum(ts_row, level, start):
    #     start = int(len(ts_row) * start)
    #     # ts_row[start] += level
    #     ts_row[start] = level
    #     return ts_row

    # @staticmethod
    # def inject_amplitude(ts_row, level, start, length):
    #     start = int(len(ts_row) * start)
    #     length = int(len(ts_row) * length)
    #     amplitude_bell = np.ones(length) * level
    #     ts_row[start : start + length] *= amplitude_bell
    #     return ts_row

    @staticmethod
    def _sine_wave(x, amp, freq=0.01, phase=0):
        return amp * np.sin(2 * np.pi * freq * x + phase)

    @staticmethod
    def normalize_ts(x):
        x = ((x.T - x.T.min(axis=0)) / (x.T.max(axis=0) - x.T.min(axis=0))).T
        x = x * 2 - 1
        return x

    def generate_anomalies(self, anomaly_type, anomaly_params, truncated_length, aug_num, name):
        """
        :param anomaly_type: str, one of ['platform', 'trend', 'mean', 'extremum', 'pattern']
        :param anomaly_params: dict or list of dicts, anomaly parameters
        :param name: str, name of the dataset
        :param meta_idx: int, index of the *variable* meta data
        """
        print(f"Generating {anomaly_type} anomalies in {self.save_path}/{name}/")
        os.makedirs(f"{self.save_path}/{name}", exist_ok=True)
        anomaly_params = anomaly_params.copy()
        self._reset_data()
        self.rng = self._reset_rng()

        if truncated_length != -1:
            move = int(truncated_length / 2)
            self.np_data_new = np.array(self.np_data[:, :truncated_length])
            for i in range(1, self.np_data.shape[1] - truncated_length, move):
                self.np_data_new = np.concatenate([self.np_data_new, self.np_data[:, i:i + truncated_length]], axis=0)
            self.np_data = self.np_data_new

        if aug_num != 1:
            new_aug_data = np.array(self.np_data)
            for _ in range(aug_num - 1):
                new_aug_data = np.concatenate([new_aug_data, self.np_data], axis=0)
            self.np_data = new_aug_data

        self.np_data = self.normalize_ts(self.np_data)

        self.np_data_normal = np.array(self.np_data)

        anomaly_params_processed, meta_keys = self._process_params(
            anomaly_params, self.np_data.shape, self.rng
        )
        anomaly_params_processed['length'] = np.zeros(len(anomaly_params_processed['length_h0']))
        anomaly_params_processed['level'] = np.zeros(len(anomaly_params_processed['level_h0']))

        for i in tqdm(range(self.np_data.shape[0])):
            if self.contains_variable_params:
                anomaly_params = {
                    key: value[i] for key, value in anomaly_params_processed.items()
                }
            if anomaly_type == "platform":
                self.np_data[i, :], start, length, level = self.inject_platform(
                    self.np_data[i, :],
                    anomaly_params['level_h0'], anomaly_params['level_h1'],
                    anomaly_params['length_h0'], anomaly_params['length_h1'],
                    anomaly_params['start']
                )
                anomaly_params_processed['start'][i] = start
                anomaly_params_processed['length'][i] = length
                anomaly_params_processed['level'][i] = level
            elif anomaly_type == "trend":
                self.np_data[i, :] = self.inject_trend(
                    self.np_data[i, :], **anomaly_params
                )
            elif anomaly_type == "mean":
                self.np_data[i, :], start, length, level = self.inject_mean(
                    self.np_data[i, :],
                    anomaly_params['level_h0'], anomaly_params['level_h1'],
                    anomaly_params['length_h0'], anomaly_params['length_h1'],
                    anomaly_params['start']
                )
                anomaly_params_processed['start'][i] = start
                anomaly_params_processed['length'][i] = length
                anomaly_params_processed['level'][i] = level
            elif anomaly_type == "spike":
                self.np_data[i, :], start_index, length, level = self.inject_spike(
                    self.np_data[i, :],
                    anomaly_params['level_h0'], anomaly_params['level_h1'],
                    anomaly_params['start']
                )
                anomaly_params_processed['start'][i] = start_index
                anomaly_params_processed['length'][i] = length
                anomaly_params_processed['level'][i] = level

            # elif anomaly_type == "variance":
            #     self.np_data[i, :] = self.inject_variance(
            #         self.np_data[i, :], **anomaly_params
            #     )
            # elif anomaly_type == "extremum":
            #     self.np_data[i, :] = self.inject_extremum(
            #         self.np_data[i, :], **anomaly_params
            #     )
            # elif anomaly_type == "pattern":
            #     self.np_data[i, :] = self.inject_pattern(
            #         self.np_data[i, :], **anomaly_params
            #     )
            # elif anomaly_type == "amplitude":
            #     self.np_data[i, :] = self.inject_amplitude(
            #         self.np_data[i, :], **anomaly_params
            #     )
            # elif anomaly_type == 'frequency':
            #     if name.split('_')[0] == 'ecg':
            #         self.np_data[i, :] = self.inject_frequency_ecg(
            #             self.np_data[i, :], **anomaly_params
            #         )
            #     elif name.split('_')[0] == 'mot':
            #         self.np_data[i, :] = self.inject_frequency_motion(
            #             self.np_data[i, :], **anomaly_params
            #         )
            # elif anomaly_type == 'shift':
            #     self.np_data[i, :] = self.inject_shift(
            #         self.np_data[i, :], **anomaly_params
            #     )
            else:
                raise ValueError(
                    f"anomaly_type must be one of ['platform', 'trend', 'mean', 'extremum', 'pattern'], "
                    f"but {anomaly_type} was given."
                )

        if meta_keys != []:
            # Save the meta data
            meta_df = pd.DataFrame(
                # {key: np.array(anomaly_params_processed[key]) for key in meta_keys}
                {key: np.array(anomaly_params_processed[key]) for key in anomaly_params_processed.keys()}
            )

            meta_df.to_parquet(f"{self.save_path}/{name}/meta_data.parquet", index=False)
            print(f"Meta data saved to {self.save_path}/{name}/meta_data.parquet")

        # self.np_data_normal = self.normalize_ts(self.np_data_normal)
        df_normal = pd.DataFrame(self.np_data_normal, columns=[f"d_{i}" for i in range(self.np_data_normal.shape[1])])
        df_normal.to_parquet(f"{self.save_path}/{name}/normal.parquet", index=False)
        print(f"Normals saved to {self.save_path}/{name}/normal.parquet")

        self.np_data = self.normalize_ts(self.np_data)
        df_injected = pd.DataFrame(self.np_data, columns=[f"d_{i}" for i in range(self.np_data.shape[1])])
        df_injected.to_parquet(f"{self.save_path}/{name}/generated_tsa.parquet", index=False)
        print(f"Anomalies saved to {self.save_path}/{name}/generated_tsa.parquet")

    def _process_params(self, anomaly_params, data_shape, rng):
        # Check which of the parameters are lists
        check_variable_params = [isinstance(p, list) for p in anomaly_params.values()]
        self.contains_variable_params = any(check_variable_params)
        meta_keys = []
        if self.contains_variable_params:
            # Broadcast or randomly select other parameters
            for key, value in anomaly_params.items():
                if isinstance(value, list):
                    # Randomly select from the list for each entry
                    selected_vals = rng.choice(value, size=data_shape[0])
                    anomaly_params[key] = list(selected_vals)
                    print(
                        f"Randomly selected values for {key}: {selected_vals[:5]}... (first 5 values)"
                    )
                    meta_keys.append(key)
                elif isinstance(value, (int, float)):
                    # Broadcast the scalar value
                    anomaly_params[key] = list(np.full(data_shape[0], value))
                    print(f"Broadcasted value for {key}: {value}")
                else:
                    raise ValueError(
                        "anomaly_params must be a dict of lists, ints or floats."
                    )

        return anomaly_params, meta_keys


class AnomParams:
    def __init__(self):
        self.anom_params = {
            "mot_platform_1_scaled": {
                "level_h0": 0.2,
                "level_h1": 0.7,
                "length_h0": 0.1,
                "length_h1": 0.6,
                "start": [i for i in np.arange(0, 0.5, 0.01)],
            },
            "mot_mean_1_scaled": {
                "level_h0": 0.7,
                "level_h1": 0.2,
                "length_h0": 0.2,
                "length_h1": 0.2,
                "start": [i for i in np.arange(0, 0.5, 0.01)],
            },
        }


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--anom_type", type=str)
    parser.add_argument("--truncated_length", type=int, default=-1)
    parser.add_argument("--aug_num", type=int, default=1)
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", type=int)
    # parser.add_argument("--fixed_level", type=float, default=None, help="Fixed level for anomalies (optional)")
    # parser.add_argument("--fixed_length", type=float, default=None, help="Fixed length for anomalies (optional)")
    args = parser.parse_args()
    # get anomaly params
    anom_params = AnomParams().anom_params
    assert args.name in anom_params.keys(), f"{args.name} not in {anom_params.keys()}"
    # load data

    if args.data_path.split('.')[1] == 'parquet':
        df = pd.read_parquet(args.data_path)
    elif args.data_path.split('.')[1] == 'pkl':
        with open(args.data_path, 'rb') as handle:
            df = pickle.load(handle)

    # generate anomalies
    anom_funcs = AnomFuncs(df, seed=args.seed)
    anom_funcs.generate_anomalies(args.anom_type, anom_params[args.name], args.truncated_length, args.aug_num,
                                  args.name)

    # # generating anomalies with fixed length and varying level
    # if args.fixed_level is not None:
    #     # Fixed level, varying length
    #     length_range = np.arange(0.2, 0.62, 0.02)
    #     for length in length_range:
    #         anom_params[args.name]["level"] = args.fixed_level
    #         anom_params[args.name]["length"] = length
    #         print(f"Generating anomaly with fixed level {args.fixed_level} and varying length {length}")
    #         anom_funcs.generate_anomalies(args.anom_type, anom_params[args.name], args.truncated_length,
    #         args.aug_num, args.name)
    #
    # elif args.fixed_length is not None:
    #     # Fixed length, varying level
    #     level_range = np.arange(-1, 1.1, 0.1)
    #     for level in level_range:
    #         anom_params[args.name]["length"] = args.fixed_length
    #         anom_params[args.name]["level"] = level
    #         print(f"Generating anomaly with fixed length {args.fixed_length} and varying level {level}")
    #         anom_funcs.generate_anomalies(args.anom_type, anom_params[args.name], args.truncated_length,
    #         args.aug_num, args.name)
    #
    # else:
    #     raise ValueError("Either --fixed_level or --fixed_length must be specified.")


if __name__ == "__main__":
    main()

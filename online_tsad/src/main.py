import os
import yaml
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound
from train_model import train_model
from alignment import black_box_function, evaluate_specific_point
from data.custom_dataloader import get_dataloaders
from datetime import datetime
import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "32"  # Set to 25% of your 128 cores
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")

    # set seed
    pl.seed_everything(0)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_id", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--data_path", type=str, default='data/mot_mix_1_hist')
    parser.add_argument("--ckpt_name", type=str, default='a=1_mixed')
    parser.add_argument("--ckpt_monitor", type=str, default='val_loss')
    parser.add_argument("--config_path", type=str, default='configs/default.yml')
    parser.add_argument("--strategy", type=str, default='auto')
    # parser.add_argument("--trail", type=str, default='three_anomalies')
    parser.add_argument("--trail", type=str, default='six_anomalies')
    # parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--test_mode", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb", type=bool, default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # load configs
    with open(args.config_path) as cfg:
        config = yaml.safe_load(cfg)

    d_config = config["data_params"]
    m_config = config["model_params"]

    X_train = pd.read_parquet(os.path.join(args.data_path, 'train_data.parquet'))
    X_val = pd.read_parquet(os.path.join(args.data_path, 'val_data.parquet'))
    X_test = pd.read_parquet(os.path.join(args.data_path, 'test_data.parquet'))
    y_test = pd.read_parquet(os.path.join(args.data_path, 'test_meta.parquet'))
    train_dataloader, trainval_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        [X_train], [X_val], [X_test, y_test], batch_size=m_config["batch_size"])

    model = train_model(args, m_config, train_dataloader, trainval_dataloader)
    # wd, f1score = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader)

    valid_point = {'platform': {"level": 0.5, "length": 0.3}, 'mean': {"level": 0.5, "length": 0.3},
                   'spike': {"level": 15, "p": 0.03}}
    # valid_point = {'amplitude': {"level": 0.5, "length": 0.3}, 'trend': {"slope": 0.01, "length": 0.3},
    #                'variance': {"level": 0.01, "length": 0.3}}
    # valid_point = {
    #     'platform': {"level": 0.5, "length": 0.3},
    #     'mean': {"level": 0.5, "length": 0.3},
    #     'spike': {"level": 15, "p": 0.03},
    #     'amplitude': {"level": 0.5, "length": 0.3},
    #     'trend': {"slope": 0.01, "length": 0.3},
    #     'variance': {"level": 0.01, "length": 0.3}
    # }

    valid_anomaly_types = list(valid_point.keys())

    pbounds = {'platform_level': (-1.0, 1.0), 'platform_length': (0.0, 0.5),
               'mean_level': (-1.0, 1.0), 'mean_length': (0.0, 0.5),
               'spike_level': (0, 20), 'spike_p': (0.0, 1.0),
               'amplitude_level': (0, 10), 'amplitude_length': (0.0, 0.5),
               'trend_slope': (-0.01, 0.01), 'trend_length': (0.0, 0.5),
               'variance_level': (0, 0.1), 'variance_length': (0.0, 0.5)}
    acquisition_function = UpperConfidenceBound(kappa=0.3)
    optimizer = BayesianOptimization(f=black_box_function, acquisition_function=acquisition_function,
                                     pbounds=pbounds, allow_duplicate_points=True, random_state=0)
    number_of_random_search = 10
    f1_calculate_interval = 10
    wd, f1score, points = list(), list(), list()
    best_point = {'platform_level': 0, 'platform_length': 0,
                  'mean_level': 0, 'mean_length': 0,
                  'spike_level': 0, 'spike_p': 0,
                  'amplitude_level': 0, 'amplitude_length': 0,
                  'trend_slope': 0, 'trend_length': 0,
                  'variance_level': 0, 'variance_length': 0}
    best_score = {'wd': np.inf, 'f1-score': 0}
    for iter in range(1000):
        if iter < number_of_random_search:
            next_point = {k: np.round(np.random.uniform(v[0], v[1]), 4) for k, v in pbounds.items()}
        else:
            next_point = {k: np.round(v, 4) for k, v in optimizer.suggest().items()}

        should_calculate_f1 = (iter % f1_calculate_interval == 0)
        loss, f1 = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point,
                                      valid_anomaly_types, next_point, calculate_f1 = should_calculate_f1)

        print(f'iter: {iter}, wd: {loss}, f1-score: {f1}, \n'
              f'next_point: {next_point}, \n'
              f'valid_point: {valid_point}')

        wd.append(loss)
        f1score.append(f1)
        points.append(next_point)

        if loss < best_score['wd']:
            best_point = next_point
            best_score = {'wd': loss, 'f1score': f1}

        optimizer.register(params=next_point, target=-loss)

    black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point,
                       valid_anomaly_types, best_point, True)

    if len(wd) != 0 or len(f1score) != 0:
        log_dir = f'logs/training/six_anomalies'
        os.makedirs(log_dir, exist_ok=True)
        with open(f'{log_dir}/bayes_wd_f1score_p_m_s.txt', 'w') as file:
        # with open(f'logs/training/{args.trail}/wd_f1score.txt', 'w') as file:
            file.write('wd: ' + str(wd))
            file.write("\n")
            file.write('f1score: ' + str(f1score))
            file.write("\n")
            file.write('points: ' + str(points))
            file.write("\n")
            file.write('best_point: ' + str(best_point))
            file.write("\n")
            file.write('best_score: ' + str(best_score))

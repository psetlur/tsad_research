import os
import yaml
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from bayes_opt import BayesianOptimization


from train_model import train_model
from alignment import black_box_function 
from data.custom_dataloader import get_dataloaders
from datetime import datetime
import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")

    pl.seed_everything(0)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_id", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--data_path", type=str, default='data/mot_mix_1_hist')
    parser.add_argument("--ckpt_name", type=str, default='a=1_mixed')
    parser.add_argument("--ckpt_monitor", type=str, default='val_loss')
    parser.add_argument("--config_path", type=str, default='configs/default.yml')
    parser.add_argument("--strategy", type=str, default='auto')
    parser.add_argument("--trail", type=str, default='six_anomalies_dynamic')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--test_mode", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb", type=bool, default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

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

    stak_params = {
        "base_kappa": 0.3,
        "boosted_kappa": 3.0,
        "stagnation_threshold": 15,
        "boost_duration": 10,
        "cooldown_factor": 0.95
    }
    print(f"--- Using STAK-UCB Strategy with params: {stak_params} ---")

    valid_point = {'amplitude': {"level": 0.5, "length": 0.3}, 'trend': {"slope": 0.01, "length": 0.3},
                   'variance': {"level": 0.01, "length": 0.3}}
    valid_anomaly_types = list(valid_point.keys())

    pbounds = {'platform_level': (-1.0, 1.0), 'platform_length': (0.0, 0.5),
               'mean_level': (-1.0, 1.0), 'mean_length': (0.0, 0.5),
               'spike_level': (0, 20), 'spike_p': (0.0, 1.0),
               'amplitude_level': (0, 10), 'amplitude_length': (0.0, 0.5),
               'trend_slope': (-0.01, 0.01), 'trend_length': (0.0, 0.5),
               'variance_level': (0, 0.1), 'variance_length': (0.0, 0.5)}

    optimizer = BayesianOptimization(f=None,
                                     pbounds=pbounds,
                                     allow_duplicate_points=True,
                                     random_state=0)

    number_of_random_search = 10
    f1_calculate_interval = 10
    wd_history, f1score_history, points_history, kappa_history = list(), list(), list(), list()
    best_point = None
    best_score = {'wd': np.inf, 'f1score': None}

    # STAK-UCB State
    last_improvement_iter = 0
    current_kappa = stak_params["base_kappa"]
    boost_active_counter = 0
    tolerance = 1e-6

    total_iterations = 500
    for iter in range(total_iterations):
        print(f"\n--- Iteration {iter}/{total_iterations} ---")
        kappa_to_use_now = current_kappa 

        if iter >= number_of_random_search:
            stagnation_duration = iter - last_improvement_iter

            if boost_active_counter > 0:
                kappa_to_use_now = stak_params["boosted_kappa"]
                print(f"  Increased Kappa, ({boost_active_counter} left). Kappa = {kappa_to_use_now:.4f}")
                boost_active_counter -= 1
                current_kappa = stak_params["boosted_kappa"]
            else:
                if stagnation_duration >= stak_params["stagnation_threshold"]:
                    kappa_to_use_now = stak_params["boosted_kappa"]
                    boost_active_counter = stak_params["boost_duration"] - 1
                    print(f" Stagnation for ({stagnation_duration} iters). New Kappa = {kappa_to_use_now:.4f}")
                    current_kappa = stak_params["boosted_kappa"]
                else:
                    cooled_kappa = current_kappa * stak_params["cooldown_factor"]
                    current_kappa = max(stak_params["base_kappa"], cooled_kappa)
                    kappa_to_use_now = current_kappa
                    

            kappa_history.append(kappa_to_use_now)
        else:
             kappa_to_use_now = stak_params["base_kappa"]
             kappa_history.append(kappa_to_use_now)
             print(f"  Random Search Phase")

        if iter < number_of_random_search:
            next_point = {k: np.round(np.random.uniform(v[0], v[1]), 4) for k, v in pbounds.items()}
        else:
             if not optimizer._space.empty:
                 next_point = optimizer.suggest(kind='ucb', kappa=kappa_to_use_now)
                 next_point = {k: np.round(v, 4) for k, v in next_point.items()}
             else:
                 print("Warning: Optimizer space empty despite being past random search. Using random point.")
                 next_point = {k: np.round(np.random.uniform(v[0], v[1]), 4) for k, v in pbounds.items()}

        should_calculate_f1 = (iter % f1_calculate_interval == 0) or (iter == total_iterations - 1) or (loss < best_score['wd'] - tolerance)
        loss, f1 = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point,
                                      valid_anomaly_types, next_point, calculate_f1 = should_calculate_f1)


        wd_history.append(loss)
        f1score_history.append(f1)
        points_history.append(next_point) 

        optimizer.register(params=next_point, target=-loss)

        if loss < best_score['wd'] - tolerance:
            best_score['wd'] = loss
            best_score['f1score'] = f1
            best_point = next_point
            last_improvement_iter = iter 

    print("\n--- Optimization Finished ---")
    final_loss, final_f1 = np.inf, None
    if best_point is not None:
        print(f"Best point found during search: {best_point}")
        print(f"Best score during search (WD): {best_score['wd']:.5f}, F1 (at that iter): {best_score['f1score'] if best_score['f1score'] is not None else 'N/A'}")
        print("\nRe-evaluating best point found for final F1 score...")
        final_loss, final_f1 = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point,
                                                   valid_anomaly_types, best_point, calculate_f1=True)
        print(f"  Re-evaluation - WD: {final_loss:.5f}, F1: {final_f1:.4f}")
        best_score['f1score'] = final_f1 
    else:
        print("No best point found (optimization might not have improved).")

    if len(wd_history) > 0:
        log_dir = f'logs/training/{args.trail}'
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = f'{log_dir}/bayes_opt_results.txt'
        print(f"\nSaving results to: {log_file_path}")
        with open(log_file_path, 'w') as file:
            file.write(f"STAK-UCB Parameters: {stak_params}\n\n")
            file.write(f"Best Point Found: {best_point}\n")
            file.write(f"Best Score Re-evaluated (WD, F1): {{'wd': {final_loss:.5f}, 'f1score': {final_f1:.4f}}}\n\n" if best_point is not None else "Best Score Re-evaluated: N/A\n\n")
            file.write(f"WD History ({len(wd_history)}): {wd_history}\n\n")
            file.write(f"F1 Score History ({len(f1score_history)}): {f1score_history}\n\n")
            file.write(f"Kappa History ({len(kappa_history)}): {kappa_history}\n\n")
            file.write(f"Points History ({len(points_history)}): {points_history}\n\n")

    print("--- Script Finished ---")
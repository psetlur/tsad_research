import os
import yaml
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound
from tqdm import tqdm
from scipy.spatial.distance import cdist

from train_model import train_model
from alignment import black_box_function
from data.custom_dataloader import get_dataloaders
from datetime import datetime
import numpy as np
import logging

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"


# Function to calculate average uncertainty
def get_average_uncertainty(optimizer, pbounds, n_samples=2000):
    """Calculates average GP predictive std dev over random samples."""
    if not hasattr(optimizer, '_gp') or optimizer._gp is None or optimizer._space.empty:
        if not optimizer._space.empty:
            try:
                optimizer._prime_subscriptions()
                if optimizer._gp is None:
                    logging.warning("GP not initialized after priming, returning default uncertainty.")
                    return 1.0
            except Exception as e:
                logging.warning(f"Error priming optimizer, returning default uncertainty: {e}")
                return 1.0
        else:
            logging.warning("Optimizer space empty, cannot calculate uncertainty yet.")
            return 1.0

    param_keys = list(pbounds.keys())
    samples_list = []
    for _ in range(n_samples):
        sample = {k: np.random.uniform(pbounds[k][0], pbounds[k][1]) for k in param_keys}
        try:
            # Use optimizer's key order if available
            ordered_sample = [sample[k] for k in optimizer.space.keys]
        except AttributeError:
            # Fallback to pbounds order if space.keys not available early on
            ordered_sample = [sample[k] for k in param_keys]
        samples_list.append(ordered_sample)
    points = np.array(samples_list)

    try:
        _, std_dev = optimizer._gp.predict(points, return_std=True)
        avg_std_dev = np.mean(std_dev)
        return max(avg_std_dev, 1e-9)
    except Exception as e:
        logging.error(f"Error during GP prediction for uncertainty: {e}")
        return 1.0


# --- main script ---
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
    # parser.add_argument("--data_path", type=str, default='data/mot_mix_1_hist')
    parser.add_argument("--data_path", type=str, default='data/ucr')
    parser.add_argument("--ckpt_name", type=str, default='a=1_mixed')
    parser.add_argument("--ckpt_monitor", type=str, default='val/loss')
    parser.add_argument("--config_path", type=str, default='configs/default.yml')
    parser.add_argument("--strategy", type=str, default='auto')
    # parser.add_argument("--trail", type=str, default='fixed_encoder')
    # parser.add_argument("--trail", type=str, default='platform1')
    parser.add_argument("--trail", type=str, default='ucr')
    # parser.add_argument("--trail", type=str, default='six_anomalies_warmup_adaptive_v5') # Updated trail name
    parser.add_argument("--device", type=str, default='auto')
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
    # wd, f1 = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader)
    # with open(f'logs/training/{args.trail}/wd_f1score.txt', 'w') as file:
    #     file.write('wd: ' + str(wd))
    #     file.write("\n")
    #     file.write('f1score: ' + str(f1))
    # raise Exception()

    # # --- Adaptive Kappa Schedule Parameters with Warmup ---
    # number_of_random_search = 10  # Pure random sampling (part of warmup)
    # schedule_params = {
    #     "n_warmup": 30,  # Total iterations for initial high exploration
    #     "kappa_warmup": 4.0,  # High kappa during warmup
    #     "kappa_adaptive_base": 0.3,  # Base kappa (kappa_0) for adaptive phase
    #     "kappa_boost": 5.0,  # Kappa when boosting due to stagnation
    #     "stagnation_threshold": 10,  # Patience before boosting
    #     "boost_duration": 5,  # How long the boost lasts
    #     "n_uncertainty_samples": 2000,  # Samples for avg uncertainty calc
    #     "kappa_min": 0.1,  # Minimum allowed adaptive kappa
    #     "kappa_max": 5.0  # Maximum allowed adaptive kappa
    # }
    # print(f"--- Using Warmup + Adaptive Kappa Schedule with params: {schedule_params} ---")
    # if schedule_params["n_warmup"] < number_of_random_search:
    #     logging.warning(
    #         f"Warmup iterations ({schedule_params['n_warmup']}) should be >= random search iterations ("
    #         f"{number_of_random_search}).")
    # # --- End Parameters ---

    # valid_point = {  # Define the target configuration
    #     'platform': {"level": 0.5, "length": 0.3}, 'mean': {"level": 0.5, "length": 0.3},
    #     'spike': {"level": 15, "p": 0.03}, 'amplitude': {"level": 0.5, "length": 0.3},
    #     'trend': {"slope": 0.01, "length": 0.3}, 'variance': {"level": 0.01, "length": 0.3}
    # }
    # valid_anomaly_types = list(valid_point.keys())

    # # best_point = {'platform_level': 0.5, 'platform_length': 0.3}
    # # wd, f1 = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point,
    # #                             valid_anomaly_types,
    # #                             best_point, calculate_f1=True)  # Calculate F1 NOW
    # # print(f'{wd}, {f1}')
    # # raise Exception()

    # pbounds = {'platform_level': (-1.0, 1.0), 'platform_length': (0.0, 0.5),
    #            'mean_level': (-1.0, 1.0), 'mean_length': (0.0, 0.5),
    #            'spike_level': (0, 20), 'spike_p': (0.0, 1.0),
    #            'amplitude_level': (0, 10), 'amplitude_length': (0.0, 0.5),
    #            'trend_slope': (-0.01, 0.01), 'trend_length': (0.0, 0.5),
    #            'variance_level': (0, 0.1), 'variance_length': (0.0, 0.5)}

    # optimizer = BayesianOptimization(f=None, pbounds=pbounds, allow_duplicate_points=True, random_state=0, verbose=0)

    # # Start with warmup kappa
    # acq_func_instance = UpperConfidenceBound(kappa=schedule_params["kappa_warmup"])
    # optimizer._acquisition_function = acq_func_instance

    # # History tracking
    # wd_history, f1score_history, points_history, kappa_history, min_1nn_dist_history, sigma_avg_history = [], [], [],\
    #     [], [], []
    # best_point = None
    # best_score = {'wd': np.inf, 'f1score': None}

    # # Schedule state variables
    # last_improvement_iter = 0
    # boost_active_counter = 0
    # sigma_avg_initial = None  # Store initial average uncertainty (after warmup)
    # tolerance = 1e-6

    # total_iterations = 500
    # for iter in tqdm(range(total_iterations), desc="Search"):
    #     kappa_to_use_now = optimizer._acquisition_function.kappa  # Start with current value

    #     # --- Determine Current Phase & Calculate Uncertainty ---
    #     is_warmup_phase = iter < schedule_params["n_warmup"]
    #     sigma_avg_current = np.nan

    #     if is_warmup_phase:
    #         # Warmup phase: Use fixed high kappa
    #         kappa_to_use_now = schedule_params["kappa_warmup"]
    #         # Calculate uncertainty on the last step of warmup to get sigma_avg_initial
    #         if iter == schedule_params["n_warmup"] - 1:
    #             if hasattr(optimizer, '_gp') and optimizer._gp is not None:
    #                 sigma_avg_current = get_average_uncertainty(optimizer, pbounds,
    #                                                             schedule_params["n_uncertainty_samples"])
    #                 sigma_avg_initial = sigma_avg_current if sigma_avg_current > 1e-9 else 1.0
    #                 tqdm.write(f"  Warmup complete. Initial Avg Uncertainty (sigma_avg(0)): {sigma_avg_initial:.4f}")
    #             else:
    #                 tqdm.write(f"  Warmup complete. Could not calculate initial uncertainty (GP not ready?).")
    #                 sigma_avg_initial = 1.0  # Use default if GP not ready
    #         # Log NaN for uncertainty during most of warmup
    #         if sigma_avg_initial is None:
    #             sigma_avg_history.append(np.nan)
    #         else:
    #             sigma_avg_history.append(sigma_avg_current)  # Log calculated value on last step

    #     else:
    #         # Adaptive phase (post-warmup)
    #         # Calculate current uncertainty
    #         if hasattr(optimizer, '_gp') and optimizer._gp is not None:
    #             sigma_avg_current = get_average_uncertainty(optimizer, pbounds,
    #                                                         schedule_params["n_uncertainty_samples"])
    #         sigma_avg_history.append(sigma_avg_current)  # Log current uncertainty (or NaN if failed)

    #         # Determine kappa based on stagnation boost or adaptive rule
    #         stagnation_duration = iter - last_improvement_iter

    #         if boost_active_counter > 0:
    #             # Currently boosting due to stagnation
    #             kappa_to_use_now = schedule_params["kappa_boost"]
    #             boost_active_counter -= 1
    #         else:
    #             # Not boosting, check for stagnation or apply adaptive rule
    #             if stagnation_duration >= schedule_params["stagnation_threshold"]:
    #                 # Start boosting
    #                 kappa_to_use_now = schedule_params["kappa_boost"]
    #                 boost_active_counter = schedule_params["boost_duration"] - 1
    #             else:
    #                 # Apply adaptive kappa based on uncertainty ratio
    #                 if sigma_avg_initial is not None and not np.isnan(sigma_avg_current):
    #                     uncertainty_ratio = sigma_avg_current / sigma_avg_initial
    #                     kappa_adaptive = schedule_params["kappa_adaptive_base"] * uncertainty_ratio
    #                     kappa_to_use_now = np.clip(kappa_adaptive, schedule_params["kappa_min"],
    #                                                schedule_params["kappa_max"])
    #                 else:
    #                     # Fallback if uncertainty not ready/available
    #                     kappa_to_use_now = schedule_params["kappa_adaptive_base"]

    #     # --- Set final kappa and log ---
    #     optimizer._acquisition_function.kappa = kappa_to_use_now
    #     kappa_history.append(kappa_to_use_now)

    #     # --- Suggest Next Point ---
    #     if iter < number_of_random_search:
    #         # Pure random sampling
    #         next_point = {k: np.round(np.random.uniform(v[0], v[1]), 4) for k, v in pbounds.items()}
    #     else:
    #         # BO suggestion using the determined kappa
    #         if not optimizer._space.empty:
    #             try:
    #                 next_point = optimizer.suggest()
    #                 next_point = {k: np.round(v, 4) for k, v in next_point.items()}
    #             except Exception as e:
    #                 logging.warning(f"Suggest failed at iter {iter}: {e}. Using random point.")
    #                 next_point = {k: np.round(np.random.uniform(v[0], v[1]), 4) for k, v in pbounds.items()}
    #         else:
    #             next_point = {k: np.round(np.random.uniform(v[0], v[1]), 4) for k, v in pbounds.items()}

    #     # --- 1NN Distance Check ---
    #     min_dist = np.nan
    #     if iter > 0 and not optimizer._space.empty and optimizer.space.params.shape[0] > 0:  # Check iter > 0
    #         try:
    #             param_keys = optimizer.space.keys
    #             next_point_arr = np.array([next_point[k] for k in param_keys]).reshape(1, -1)
    #             previous_points = optimizer.space.params

    #             distances = cdist(next_point_arr, previous_points, metric='euclidean')
    #             min_dist = np.min(distances)
    #         except Exception as e:
    #             logging.error(f"Error calculating 1NN distance at iter {iter}: {e}")
    #     min_1nn_dist_history.append(min_dist)

    #     # --- Evaluate Objective (WD only) ---
    #     should_calculate_f1_this_iter = False
    #     loss, f1 = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point,
    #                                   valid_anomaly_types, next_point,
    #                                   calculate_f1=should_calculate_f1_this_iter)

    #     # Store results
    #     wd_history.append(loss)
    #     f1score_history.append(f1)
    #     points_history.append(next_point)

    #     # Print summary periodically
    #     if (iter + 1) % 10 == 0:
    #         tqdm.write(
    #             f'  Iter {iter}: WD={loss:.5f}, Kappa={kappa_history[-1]:.4f}, SigmaAvg={sigma_avg_current:.4f}, '
    #             f'1NN_Dist={min_dist:.4f}')

    #     # Register with optimizer (Updates GP)
    #     optimizer.register(params=next_point, target=-loss)

    #     # Update best score
    #     if loss < best_score['wd'] - tolerance:
    #         tqdm.write(f"  *** Iter {iter}: New best WD: {loss:.5f} (improved from {best_score['wd']:.5f}) ***")
    #         best_score['wd'] = loss
    #         best_score['f1score'] = None
    #         best_point = next_point
    #         # Reset stagnation counter only if improvement happens AFTER warmup
    #         if not is_warmup_phase:
    #             last_improvement_iter = iter
    #         else:  # During warmup, keep resetting to start of adaptive phase
    #             last_improvement_iter = schedule_params["n_warmup"]

    # # --- Optimization Finished ---
    # print("\n--- Optimization Finished ---")
    # final_loss, final_f1 = np.inf, None

    # if best_point is not None:
    #     print(f"Best point found: {best_point}")
    #     print(f"Best WD score: {best_score['wd']:.5f}")
    #     print("\nRe-evaluating best point for final F1 score...")
    #     final_loss, final_f1 = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader,
    #                                               valid_point,
    #                                               valid_anomaly_types, best_point,
    #                                               calculate_f1=True)
    #     print(f"  Re-evaluation - WD: {final_loss:.5f}, F1: {final_f1:.4f}")
    #     best_score['f1score'] = final_f1
    # else:
    #     print("No valid best point found.")

    # # Save results
    # if len(wd_history) > 0:
    #     log_dir = f'logs/training/{args.trail}'
    #     os.makedirs(log_dir, exist_ok=True)
    #     log_file_path = f'{log_dir}/warmup_adaptive_kappa_bayes_opt_results_all.txt'
    #     print(f"\nSaving results to: {log_file_path}")
    #     with open(log_file_path, 'w') as file:
    #         file.write(f"Warmup + Adaptive Kappa Schedule Parameters: {schedule_params}\n\n")
    #         file.write(f"Best Point Found: {best_point}\n")
    #         file.write(
    #             f"Best Score Re-evaluated (WD, F1): {{'wd': {final_loss:.5f}, 'f1score': {final_f1:.4f}}}\n\n" if
    #             best_point is not None else "Best Score Re-evaluated: N/A\n\n")
    #         file.write(f"WD History ({len(wd_history)}): {wd_history}\n\n")
    #         file.write(f"Kappa History ({len(kappa_history)}): {kappa_history}\n\n")
    #         file.write(f"Avg Uncertainty History ({len(sigma_avg_history)}): {sigma_avg_history}\n\n")
    #         file.write(f"Min 1NN Distance History ({len(min_1nn_dist_history)}): {min_1nn_dist_history}\n\n")
    #         file.write(f"Points History ({len(points_history)}): {points_history}\n\n")

    # print("--- Script Finished ---")

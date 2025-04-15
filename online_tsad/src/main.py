import os
import yaml
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from bayes_opt import BayesianOptimization
# Import the acquisition function class directly
from bayes_opt.acquisition import UpperConfidenceBound

from train_model import train_model
from alignment import black_box_function
from data.custom_dataloader import get_dataloaders
from datetime import datetime
import numpy as np
import logging  # Import logging

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"

# Set logging level for BO to avoid excessive messages if desired
# logging.getLogger('bayes_opt').setLevel(logging.WARNING)

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
    # parser.add_argument("--trail", type=str, default='six_anomalies')
    parser.add_argument("--trail", type=str, default='six_anomalies_stak_ucb_fast')  # Updated trail
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
        "boosted_kappa": 0.8,
        "stagnation_threshold": 15,
        "boost_duration": 10,
        "cooldown_factor": 0.95
    }
    print(f"--- Using STAK-UCB Strategy with params: {stak_params} ---")

    # valid_point = {'amplitude': {"level": 0.5, "length": 0.3}, 'trend': {"slope": 0.01, "length": 0.3},
    #                'variance': {"level": 0.01, "length": 0.3}}
    valid_point = {'platform': {"level": 0.5, "length": 0.3}}
    valid_anomaly_types = list(valid_point.keys())

    best_point = {'platform_level': 0.5, 'platform_length': 0.3,
                  'mean_level': 0, 'mean_length': 0,
                  'spike_level': 0, 'spike_p': 0,
                  'amplitude_level': 0, 'amplitude_length': 0,
                  'trend_slope': 0, 'trend_length': 0,
                  'variance_level': 0, 'variance_length': 0}
    l, f = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point,
                              valid_anomaly_types, best_point,False,False)

    print(1)

    pbounds = {'platform_level': (-1.0, 1.0), 'platform_length': (0.0, 0.5),
               'mean_level': (-1.0, 1.0), 'mean_length': (0.0, 0.5),
               'spike_level': (0, 20), 'spike_p': (0.0, 1.0),
               'amplitude_level': (0, 10), 'amplitude_length': (0.0, 0.5),
               'trend_slope': (-0.01, 0.01), 'trend_length': (0.0, 0.5),
               'variance_level': (0, 0.1), 'variance_length': (0.0, 0.5)}
    # Initialize optimizer without f
    optimizer = BayesianOptimization(f=None,
                                     pbounds=pbounds,
                                     allow_duplicate_points=True,
                                     random_state=0,
                                     verbose=0)  # Reduce BO verbosity if desired

    # --- Fix for Issue 1: Manage Acquisition Function Instance ---
    # Create the acquisition function instance we'll modify
    # Note: xi is for EI/POI, irrelevant for UCB but needed by the class structure
    acq_func_instance = UpperConfidenceBound(kappa=stak_params["base_kappa"])
    # Assign it to the optimizer's internal attribute (use with caution)
    optimizer._acquisition_function = acq_func_instance
    # --- End Fix for Issue 1 ---

    number_of_random_search = 10
    # f1_calculate_interval = 10 # No longer needed for loop control

    wd_history, f1score_history, points_history, kappa_history = list(), list(), list(), list()
    best_point = None
    best_score = {'wd': np.inf, 'f1score': None}  # F1 starts as None

    # STAK-UCB State
    last_improvement_iter = 0
    current_kappa = stak_params["base_kappa"]  # Use initial kappa from instance
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
                print(f"  STAK: Boost active ({boost_active_counter} left). Kappa = {kappa_to_use_now:.4f}")
                boost_active_counter -= 1
                current_kappa = stak_params["boosted_kappa"]
            else:
                if stagnation_duration >= stak_params["stagnation_threshold"]:
                    kappa_to_use_now = stak_params["boosted_kappa"]
                    boost_active_counter = stak_params["boost_duration"] - 1
                    print(
                        f"  STAK: Stagnation ({stagnation_duration} iters). Boosting kappa. Kappa = "
                        f"{kappa_to_use_now:.4f}")
                    current_kappa = stak_params["boosted_kappa"]
                else:
                    cooled_kappa = current_kappa * stak_params["cooldown_factor"]
                    current_kappa = max(stak_params["base_kappa"], cooled_kappa)
                    kappa_to_use_now = current_kappa
                    # Only print if kappa actually changed significantly, or periodically
                    # if abs(kappa_to_use_now - kappa_history[-1]) > 1e-3:
                    #      print(f"  STAK: Cooldown/Stable. Kappa = {kappa_to_use_now:.4f}")

            # --- Fix for Issue 1 (continued): Modify kappa on the instance ---
            optimizer._acquisition_function.kappa = kappa_to_use_now
            # --- End Fix ---
            kappa_history.append(kappa_to_use_now)
        else:
            # Use base kappa during random search
            kappa_to_use_now = stak_params["base_kappa"]
            optimizer._acquisition_function.kappa = kappa_to_use_now  # Set it initially too
            kappa_history.append(kappa_to_use_now)
            print(f"  Random Search Phase. Kappa = {kappa_to_use_now:.4f}")

        # Suggest next point
        if iter < number_of_random_search:
            next_point = {k: np.round(np.random.uniform(v[0], v[1]), 4) for k, v in pbounds.items()}
        else:
            if not optimizer._space.empty:
                # Call suggest() without kind/kappa args, uses internal modified acq func
                next_point = optimizer.suggest()
                next_point = {k: np.round(v, 4) for k, v in next_point.items()}
            else:
                print("Warning: Optimizer space empty. Using random point.")
                next_point = {k: np.round(np.random.uniform(v[0], v[1]), 4) for k, v in pbounds.items()}

        # --- Fix for Issue 2: Control F1 Calculation ---
        # ONLY calculate F1 if explicitly needed (e.g., final eval), NOT during loop
        should_calculate_f1_this_iter = False  # Default to NOT calculating F1
        # --- End Fix ---

        # Evaluate black_box_function (returns WD, and F1=0.0 if not calculated)
        loss, f1 = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, valid_point,
                                      valid_anomaly_types, next_point,
                                      calculate_f1=should_calculate_f1_this_iter)  # Pass False here

        # Store results (f1 will be 0.0 most iterations)
        wd_history.append(loss)
        f1score_history.append(f1)  # Store the 0.0 F1
        points_history.append(next_point)
        kappa_history.append(kappa_to_use_now)  # Log kappa used for this iter's suggestion

        print(f'  Iter {iter}: WD={loss:.5f}, Kappa Used={kappa_history[-1]:.4f}')  # Removed F1 print

        # Register with optimizer (use negative loss for maximization)
        optimizer.register(params=next_point, target=-loss)

        # Update best score based on WD only
        if loss < best_score['wd'] - tolerance:
            print(f"  *** New best WD: {loss:.5f} (improved from {best_score['wd']:.5f}) ***")
            best_score['wd'] = loss
            # Don't store F1 here, as it wasn't calculated accurately
            best_score['f1score'] = None  # Mark F1 for best point as unknown for now
            best_point = next_point
            last_improvement_iter = iter

    # --- Optimization Finished ---
    print("\n--- Optimization Finished ---")
    final_loss, final_f1 = np.inf, None  # Initialize final values

    if best_point is not None:
        print(f"Best point found during search: {best_point}")
        print(f"Best WD score during search: {best_score['wd']:.5f}")

        print("\nRe-evaluating best point found to get WD and calculate final F1 score...")
        # --- Fix for Issue 2 (continued): Calculate F1 ONCE here ---
        final_loss, final_f1 = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader,
                                                  valid_point,
                                                  valid_anomaly_types, best_point,
                                                  calculate_f1=True)  # Calculate F1 NOW
        # --- End Fix ---
        print(f"  Re-evaluation - WD: {final_loss:.5f}, F1: {final_f1:.4f}")
        # Store the accurately calculated final F1 score
        best_score['f1score'] = final_f1
    else:
        print("No best point found (optimization might not have improved).")

    # Save results
    if len(wd_history) > 0:
        log_dir = f'logs/training/{args.trail}'
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = f'{log_dir}/bayes_opt_atv_results.txt'
        print(f"\nSaving results to: {log_file_path}")
        with open(log_file_path, 'w') as file:
            file.write(f"STAK-UCB Parameters: {stak_params}\n\n")
            file.write(f"Best Point Found: {best_point}\n")
            # Report the final re-evaluated scores
            file.write(
                f"Best Score Re-evaluated (WD, F1): {{'wd': {final_loss:.5f}, 'f1score': {final_f1:.4f}}}\n\n" if
                best_point is not None else "Best Score Re-evaluated: N/A\n\n")
            file.write(f"WD History ({len(wd_history)}): {wd_history}\n\n")
            # F1 history will be mostly 0.0, maybe less useful to log? Optional.
            file.write(f"F1 Score History ({len(f1score_history)}): {f1score_history}\n\n")
            file.write(f"Kappa History ({len(kappa_history)}): {kappa_history}\n\n")
            # Points history can be very large, maybe save separately if needed
            file.write(f"Points History ({len(points_history)}): {points_history}\n\n")

    print("--- Script Finished ---")

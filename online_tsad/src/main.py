import os
import yaml
import argparse
import pandas as pd
from termcolor import colored

import torch
import pytorch_lightning as pl

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement, UpperConfidenceBound

from train_model import train_model
from alignment import black_box_function
from data.custom_dataloader import get_dataloaders
from datetime import datetime

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
    # parser.add_argument("--trail", type=str, default='fixed')
    # parser.add_argument("--trail", type=str, default='grid')
    # parser.add_argument("--trail", type=str, default='more_epochs')
    # parser.add_argument("--trail", type=str, default='second_loss')
    # parser.add_argument("--trail", type=str, default='length_optimized')
    # parser.add_argument("--trail", type=str, default='more_negative')
    # parser.add_argument("--trail", type=str, default='warmup')
    # parser.add_argument("--trail", type=str, default='second_anomaly')
    parser.add_argument("--trail", type=str, default='inject_spike')
    # parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument(
        "--test_mode", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use wandb for logging",
    )
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
        [X_train],
        [X_val],
        [X_test, y_test],
        batch_size=m_config["batch_size"],
    )

    # pbounds = {
    #     "ratio_0"  : (0.01, 0.2), "ratio_1"  : (0.01, 0.2),
    #     "length_0_h0": (0.001, 1), "length_0_h1": (0.001, 1),
    #     "level_0_h0" : (0.001, 1), "level_0_h1" : (0.001, 1),
    #     "length_1_h0": (0.001, 1), "length_1_h1": (0.001, 1),
    #     "level_1_h0" : (0.001, 1), "level_1_h1" : (0.001, 1),
    # }
    # best_point = {
    #     "ratio_0"  : 0.05, "ratio_1"  : 0.05,
    #     "length_0_h0": 0.1, "length_0_h1": 0.6,
    #     "level_0_h0" : 0.2, "level_0_h1" : 0.7,
    #     "length_1_h0": 0.2, "length_1_h1": 0.2,
    #     "level_1_h0" : 0.7, "level_1_h1" : 0.2,
    # }
    best_point = {
        "ratio_anomaly": 0.1,  # Total ratio of anomalies (e.g., 10% of the data)
        "fixed_level": 0.5,  # Fixed level for the platform anomaly
        "fixed_length": 0.3,  # Fixed length for the platform anomaly
        "fixed_start": 0.2  # Fixed start position for the platform anomaly
    }

    # num_iterations = 5
    # for iteration in range(num_iterations):
    model = train_model(args, m_config, train_dataloader, trainval_dataloader, best_point)
    wd, f1score = black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, best_point)
    # print(colored("Ground truth point:", 'blue'), best_point)
    # print(colored(f"Iteration {iteration + 1} - Target Value:", 'blue'), target)
    # print(colored(f"Iteration {iteration + 1} - F1-Score   :", 'blue'), f1score)
    print(colored(f"WD:", 'blue'), wd)
    print(colored(f"F1-Score:", 'blue'), f1score)
    print()
    if len(wd) != 0 or len(f1score) != 0:
        with open(f'logs/training/{args.trail}/wd_f1score.txt', 'w') as file:
            file.write('wd: ' + str(wd))
            file.write("\n")
            file.write('f1score: ' + str(f1score))

    # acquisition_function = UpperConfidenceBound(kappa=0.1)
    # optimizer = BayesianOptimization(
    #     f=black_box_function,
    #     acquisition_function=acquisition_function,
    #     pbounds=pbounds,
    #     allow_duplicate_points=True,
    #     random_state=0,
    # )

    # best_target = -np.inf
    # number_of_random_search = 10
    # for iter in range(5):
    #     print("Iteration", iter)

    #     if iter < number_of_random_search:
    #         next_point_to_probe = {k: np.round(np.random.uniform(v[0], v[1]), 3) for k, v in pbounds.items()}
    #     else:
    #         next_point_to_probe = {k: np.round(v, 3) for k, v in optimizer.suggest().items()}
    #     print("Next point to probe is:", next_point_to_probe)

    #     a_config = {
    #         "ratio_anomaly": best_point["ratio_anomaly"],
    #         "fixed_level": best_point["fixed_level"],
    #         "fixed_length": best_point["fixed_length"],
    #         "fixed_start": best_point["fixed_start"]
    #     }

    #     if a_config["fixed_length"] > 1 or a_config["ratio_anomaly"] > 1:
    #         target, f1score = -10, 0
    #     else:
    #         model = train_model(args, m_config, train_dataloader, trainval_dataloader, a_config)
    #         target, f1score = black_box_function(model, train_dataloader, val_dataloader, test_dataloader, a_config)

    #     ### Assign a large negative score to illegal hyperparameters, i.e., with probability sum larger than 1
    #     # if next_point_to_probe['length_0_h0'] + next_point_to_probe['length_0_h1'] > 0.999 or \
    #     #     next_point_to_probe['length_1_h0'] + next_point_to_probe['length_1_h1'] > 0.999 or \
    #     #     next_point_to_probe['level_0_h0'] + next_point_to_probe['level_0_h1'] > 0.999 or \
    #     #     next_point_to_probe['level_1_h0'] + next_point_to_probe['level_1_h1'] > 0.999:
    #     #     target, f1score = -10, 0
    #     # else:
    #     #     model = train_model(args, m_config, train_dataloader, trainval_dataloader, next_point_to_probe)
    #     #     target, f1score = black_box_function(model, train_dataloader, val_dataloader, test_dataloader,
    #     next_point_to_probe)
    #     print("Found the target value to be:", target)
    #     print("Test F1-Score:", f1score)

    #     if target > best_target:
    #         best_target = target
    #         print(colored('Best!', 'red'))

    #     optimizer.register(
    #         params=next_point_to_probe,
    #         target=target,
    #     )
    #     print()

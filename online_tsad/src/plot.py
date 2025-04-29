import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import os

trail = 'platform1'


def plot_loss_curve(last=False):
    df = pd.read_csv(f'logs/training/{trail}/metrics.csv').dropna(subset=["val_loss"])
    if last is True:
        df = df.tail(100)
        skip = 20
    else:
        skip = 100
    epoch = df["epoch"]
    plt.figure(figsize=(6, 4))
    plt.plot(epoch, df["loss_global"], marker="s", markersize=10, markerfacecolor='none', linestyle="-",
             markeredgecolor="red", color="red", label=f"Outlier-Outlier", markevery=skip)
    plt.plot(epoch, df["loss_local"], marker="^", markersize=10, markerfacecolor='none', linestyle="-",
             markeredgecolor="blue", color="blue", label=f"Continuous-HP", markevery=skip)
    plt.plot(epoch, df["loss_normal"], marker="v", markersize=10, markerfacecolor='none', linestyle="-",
             markeredgecolor="green", color="green", label=f"Inlier-Outlier", markevery=skip)
    plt.plot(epoch, df["val_loss"], marker="o", markersize=10, markerfacecolor='none', linestyle="-",
             markeredgecolor="black", color="black", label=f"Overall Loss", markevery=skip)
    plt.xticks(np.arange(epoch.iloc[0], epoch.iloc[-1], skip))
    plt.title("Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'logs/training/{trail}/loss_curve_last{last}.pdf')
    plt.show()
    plt.close()


def plot_wd_f1score():
    with open(f"logs/training/{trail}/wd_f1score.txt", "r") as f:
        lines = f.readlines()
        wd = ast.literal_eval(lines[0][4:])
        f1score = ast.literal_eval(lines[1][9:])

    def plot_heatmap(data, title, config_name):
        configs = sorted(data.keys())
        n = len(configs)
        x_indices, y_indices = np.meshgrid(range(n), range(n))
        values = np.zeros((n, n))
        for i, x in enumerate(configs):
            for j, y in enumerate(configs):
                values[i, j] = data[x][y]
        plt.figure(figsize=(8, 6))
        mesh = plt.pcolormesh(y_indices, x_indices, np.ma.masked_where(values == -1, values),
                              cmap="viridis", vmin=np.min(values), vmax=np.max(values))
        if title[-2:] == 'WD':
            for i in range(n):
                column = values[:, i]
                min_idx = np.argmin(column)
                plt.scatter(i, min_idx, color='red', s=50, edgecolor='black', label='Min Value')
        else:
            for i in range(n):
                column = values[:, i]
                max_idx = np.argmax(column)
                plt.scatter(i, max_idx, color='red', s=50, edgecolor='black', label='Max Value')
        plt.xticks(range(n), configs, rotation=90)
        plt.yticks(range(n), configs)
        plt.colorbar(mesh, label='Value')
        plt.xlabel(f'train_{config_name}')
        plt.ylabel(f'test_{config_name}')
        plt.title(title)
        plt.tight_layout()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')
        plt.savefig(f'logs/training/{trail}/{config_name}_{title}.pdf')
        plt.show()

    for anomaly_type in wd.keys():
        for config in wd[anomaly_type].keys():
            plot_heatmap(data=wd[anomaly_type][config], title=f'{anomaly_type} WD', config_name=config)
            plot_heatmap(data=f1score[anomaly_type][config], title=f'{anomaly_type} F1-score', config_name=config)


def _plot_wd_f1score():
    with open(f"logs/training/{trail}/wd_f1score.txt", "r") as f:
        lines = f.readlines()
        wd = ast.literal_eval(lines[0][4:])
        # f1score = ast.literal_eval(lines[1][9:])

    def plot_heatmap(data, title, config_name):
        configs = sorted(data.keys())
        n = len(configs)
        x_indices, y_indices = np.meshgrid(range(n), range(n))
        values = np.zeros((n, n))
        for i, x in enumerate(configs):
            for j, y in enumerate(configs):
                values[i, j] = data[x][y]
        plt.figure(figsize=(8, 6))
        mesh = plt.pcolormesh(y_indices, x_indices, np.ma.masked_where(values == -1, values),
                              cmap="viridis", vmin=np.min(values), vmax=np.max(values))
        if title[-2:] == 'WD':
            for i in range(n):
                column = values[i, :]
                min_idx = np.argmin(column)
                plt.scatter(i, min_idx, color='red', s=50, edgecolor='black', label='Min Value')
        else:
            for i in range(n):
                column = values[i, :]
                reversed_column = column[::-1]
                max_idx_reversed = np.argmax(reversed_column)
                max_idx = len(column) - 1 - max_idx_reversed
                plt.scatter(i, max_idx, color='red', s=50, edgecolor='black', label='Max Value')
        plt.xticks(range(n), configs, rotation=90)
        plt.yticks(range(n), configs)
        plt.colorbar(mesh, label='Value')
        plt.xlabel(f'train_{config_name}')
        plt.ylabel(f'test_{config_name}')
        plt.title(title)
        plt.tight_layout()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')
        plt.savefig(f'logs/training/{trail}/{config_name}_{title}.pdf')
        plt.show()

    anomaly_types = ['platform']
    # anomaly_types = ['spike']

    for anomaly_type in anomaly_types:
        for config in wd[anomaly_type].keys():
            plot_heatmap(data=wd[anomaly_type][config], title=f'{anomaly_type} WD', config_name=config)
            # plot_heatmap(data=f1score[anomaly_type][config], title=f'{anomaly_type} F1-score', config_name=config)


def plot_classifier_loss_curve(config):
    with open(f"logs/training/{trail}/classifier_loss_{config}.txt", "r") as f:
        lines = f.readlines()
        loss_data = ast.literal_eval(lines[0])

    epochs = range(1, len(loss_data) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_data, marker='o', linestyle='-', color='b', label='Loss', markevery=100)
    plt.title('Validation Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.savefig(f'logs/training/{trail}/classifier_loss_{config}.pdf')


def plot_wd_f1score_combined():
    with open(f"logs/training/{trail}/wd_f1score.txt", "r") as f:
        lines = f.readlines()
        wd = ast.literal_eval(lines[0][4:])
        f1score = ast.literal_eval(lines[1][9:])

    def plot_heatmap(data, title):
        axis = list(data.keys())
        num = np.arange(len(axis))
        values = np.zeros((len(axis), len(axis)))
        for i, x in enumerate(axis):
            for j, y in enumerate(axis):
                values[i, j] = data[x][y]
        axis = [f"{str(key)}/ {i}" for i, key in enumerate(axis)]
        plt.figure(figsize=(24, 18))
        plt.pcolormesh(num, num, values, cmap="viridis", vmin=np.min(values), vmax=np.max(values))
        if title[-2:] == 'WD':
            for i in range(values.shape[1]):
                column = values[i, :]
                min_index = np.argmin(column)
                plt.scatter(i, min_index, color='red', s=50, edgecolor='black', label='Min Value')
        else:
            for i in range(values.shape[1]):
                column = values[i, :]
                max_index = np.argmax(column)
                plt.scatter(i, max_index, color='red', s=50, edgecolor='black', label='Max Value')

        plt.xticks(ticks=num)
        plt.yticks(ticks=num, labels=axis)
        plt.colorbar(label='Value')
        plt.xlabel(f'train')
        plt.ylabel(f'test')
        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')
        plt.savefig(f'logs/training/{trail}/{title}.pdf')
        plt.show()

    plot_heatmap(data=wd, title=f'WD')
    plot_heatmap(data=f1score, title=f'F1-Score')


def plot_wd_f1score_line(input_filepath, output_filepath, type="wd", anomaly="platform", level=0.5, length=0.3,
                         best_value=None, baseline=None):
    df = pd.read_csv(input_filepath)
    plt.figure(figsize=(12, 10))

    # Map 'f1' to 'f1score' to match CSV column name
    column_name = "f1score" if type == "f1" else type

    # For WD type, handle scaling issues
    if type == "wd":
        # Calculate running minimum WD for each iteration
        df['running_best'] = df[column_name].cummin()
        best_label = 'Running Min WD'

        # Highlight global minimum
        global_best = df[column_name].min()
        global_idx = df[column_name].idxmin()
        global_iter = df.loc[global_idx, 'iter']
        global_label = 'Global Min WD'

        # Determine appropriate y-axis limit
        # Get the 75th percentile or a reasonable max value to focus on
        y_max = min(df[column_name].quantile(0.75) * 1.5, df[column_name].median() * 5)

        # If we have extreme outliers, cap the values for visualization purposes
        df['capped_values'] = df[column_name].clip(upper=y_max)

        # Plot capped values
        plt.plot(df['iter'], df['capped_values'], marker='o', linestyle='-', color='blue',
                 linewidth=2, label='WD Values (capped)')

        # Mark any capped values
        capped_indices = df[df[column_name] > y_max].index
        if not capped_indices.empty:
            plt.scatter(df.loc[capped_indices, 'iter'],
                        [y_max] * len(capped_indices),
                        marker='^', color='purple', s=100,
                        label=f'Values exceeding {y_max:.2f}')
    else:
        # For F1 score, plot normally
        plt.plot(df['iter'], df[column_name], marker='o', linestyle='-', color='blue',
                 linewidth=2, label='Actual Values')

        # Calculate running maximum F1 for each iteration
        df['running_best'] = df[column_name].cummax()
        best_label = 'Running Max F1'

        # Highlight global maximum
        global_best = df[column_name].max()
        global_idx = df[column_name].idxmax()
        global_iter = df.loc[global_idx, 'iter']
        global_label = 'Global Max F1'

    # Plot running best as a separate line
    if type == "wd":
        # For WD, cap the running best too
        plt.plot(df['iter'], df['running_best'].clip(upper=y_max), marker='o', linestyle='--',
                 color='green', linewidth=1.5, label=best_label)
    else:
        plt.plot(df['iter'], df['running_best'], marker='o', linestyle='--',
                 color='green', linewidth=1.5, label=best_label)

    # Highlight global best with a star
    if type == "wd" and global_best > y_max:
        # If global best is beyond the cap, place the star at the cap
        plt.scatter(global_iter, y_max, color='red', s=500, marker='*',
                    zorder=5, label=f'{global_label} ({global_best:.4f})')
    else:
        plt.scatter(global_iter, global_best, color='red', s=500, marker='*',
                    zorder=5, label=global_label)

    # Add the horizontal line for the best value if provided
    if best_value is not None:
        best_type = "Best WD" if type == "wd" else "Best F1"
        if type == "wd" and best_value > y_max:
            # If best value exceeds the cap, don't show the line
            plt.text(0.02, 0.98, f'{best_type} = {best_value:.4f} (beyond scale)',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
        else:
            plt.axhline(y=best_value, color='red', linestyle='-', linewidth=2,
                        label=f'{best_type} = {best_value:.4f}')

    # Adding horizontal line for baseline
    if baseline is not None:
        baseline_type = "Baseline WD" if type == "wd" else "Baseline F1"
        if type == "wd" and baseline > y_max:
            # If baseline exceeds the cap, don't show the line
            plt.text(0.02, 0.92, f'{baseline_type} = {baseline:.4f} (beyond scale)',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
        else:
            plt.axhline(y=baseline, color='purple', linestyle='-', linewidth=2,
                        label=f'{baseline_type} = {baseline:.4f}')

    plt.xlabel("Iteration")
    plt.ylabel(type.upper())
    plt.title(f"{type.upper()} for {anomaly} anomalies (level={level}, length={length})")

    if type == "wd":
        plt.text(0.5, 0.01, f"Note: Y-axis capped at {y_max:.2f} to focus on smaller WD values",
                 ha='center', transform=plt.gcf().transFigure, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.1))

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.show()


def plot_level_length_changes(input_filepath, output_filepath, level=0.5, length=0.3, spike_level=15, spike_p=0.03,
                              anomaly="platform"):
    # Read the CSV file
    df = pd.read_csv(input_filepath)

    # Create a figure with four subplots
    fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # Plot 1: Platform & Mean Levels only (separated from spike)
    axs[0].plot(df['iter'], df['platform_level'], marker='.', linestyle='-',
                color='blue', linewidth=2, label='Platform Level')
    axs[0].plot(df['iter'], df['mean_level'], marker='.', linestyle='-',
                color='orange', linewidth=2, label='Mean Level')
    axs[0].axhline(y=level, color='red', linestyle='-', linewidth=3, alpha=1.0,
                   label=f'Target Level ({level})')
    axs[0].set_ylabel('Level')
    axs[0].set_title('Platform & Mean Level Changes Over Iterations')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(loc='best')

    # Plot 2: Spike Level (separate)
    axs[1].plot(df['iter'], df['spike_level'], marker='.', linestyle='-',
                color='green', linewidth=2, label='Spike Level')
    axs[1].axhline(y=spike_level, color='darkgreen', linestyle='-', linewidth=3, alpha=1.0,
                   label=f'Target Spike Level ({spike_level})')
    axs[1].set_ylabel('Level')
    axs[1].set_title('Spike Level Changes Over Iterations')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend(loc='best')

    # Plot 3: Platform & Mean Length (combined)
    axs[2].plot(df['iter'], df['platform_length'], marker='.', linestyle='-',
                color='blue', linewidth=2, label='Platform Length')
    axs[2].plot(df['iter'], df['mean_length'], marker='.', linestyle='-',
                color='orange', linewidth=2, label='Mean Length')
    axs[2].axhline(y=length, color='red', linestyle='-', linewidth=3, alpha=1.0,
                   label=f'Target Length ({length})')
    axs[2].set_ylabel('Length')
    axs[2].set_title('Platform & Mean Length Changes Over Iterations')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend(loc='best')

    # Plot 4: Spike Probability (separate)
    axs[3].plot(df['iter'], df['spike_p'], marker='.', linestyle='-',
                color='green', linewidth=2, label='Spike Probability')
    axs[3].axhline(y=spike_p, color='darkgreen', linestyle='-', linewidth=3, alpha=1.0,
                   label=f'Target Probability ({spike_p})')
    axs[3].set_xlabel('Iteration')
    axs[3].set_ylabel('Probability')
    axs[3].set_title('Spike Probability Changes Over Iterations')
    axs[3].grid(True, linestyle='--', alpha=0.7)
    axs[3].legend(loc='best')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.show()

    return fig


def plot_wd_f1score_spike():
    with open(f"logs/training/{trail}/spike_wd_f1score.txt", "r") as f:
        lines = f.readlines()
        wd = ast.literal_eval(lines[0][4:])
        f1score = ast.literal_eval(lines[1][9:])

    def plot_heatmap(data, title, config_name):
        configs = np.round(np.arange(0.1, 1.1, 0.1), 1)
        x_config, y_config = np.meshgrid(configs, configs)

        x_values = sorted(data.keys())
        y_values = sorted(data[x_values[0]].keys())
        values = np.zeros((len(y_values), len(x_values)))
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                values[j, i] = data[x][y]
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(y_config, x_config, values, cmap="viridis", vmin=np.min(values), vmax=np.max(values))
        # plt.pcolormesh(y_config, x_config, np.ma.masked_where(values == 0, values), cmap="viridis",
        #                vmin=np.min(values), vmax=np.max(values))
        # if title[-2:] == 'WD':
        #     for i in range(values.shape[1]):
        #         column = values[:, i]
        #         min_index = np.argmin(column)
        #         plt.scatter(x_config[min_index, i], y_config[min_index, i], color='red', s=50, edgecolor='black',
        #                     label='Min Value')
        # else:
        #     for i in range(values.shape[1]):
        #         column = values[:, i]
        #         max_index = np.argmax(column)
        #         plt.scatter(x_config[max_index, i], y_config[max_index, i], color='red', s=50, edgecolor='black',
        #                     label='Max Value')
        plt.xticks(configs, rotation=90)
        plt.yticks(configs)
        plt.colorbar(label='Value')
        plt.xlabel(f'train_{config_name}')
        plt.ylabel(f'test_{config_name}')
        plt.title(title)
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys(), loc='upper left')
        plt.savefig(f'logs/training/{trail}/{title}.pdf')
        plt.show()

    plot_heatmap(data=wd, title=f'spike_WD', config_name='p')
    plot_heatmap(data=f1score, title=f'spike_F1-score', config_name='p')


def plot_level_length_step_function(input_filepath, output_filepath, level=0.5, length=0.3, spike_level=15,
                                    spike_p=0.03):
    # Read the CSV file
    df = pd.read_csv(input_filepath)

    # Track minimum WD seen so far at each iteration
    df['min_wd_so_far'] = df['wd'].cummin()

    # Create a mask for rows where the min_wd changes
    is_new_min = df['min_wd_so_far'] != df['min_wd_so_far'].shift(1)
    is_new_min.iloc[0] = True  # First row is always a new minimum

    # Create a dataframe to store the best configuration at each iteration
    best_config = pd.DataFrame(index=df.index)
    best_config['iter'] = df['iter']

    # Initialize with values from the first row
    current_best_idx = 0

    # Parameters to track
    params = ['platform_level', 'platform_length', 'mean_level',
              'mean_length', 'spike_level', 'spike_p']

    # For each parameter, fill with the best value seen so far
    for param in params:
        best_config[param] = None

    # Fill the best_config dataframe with step function values
    for i in range(len(df)):
        if is_new_min.iloc[i]:
            current_best_idx = i

        for param in params:
            best_config.loc[i, param] = df.loc[current_best_idx, param]

    # Create a figure with four subplots
    fig, axs = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

    # Plot 1: Platform & Mean Levels
    axs[0].plot(best_config['iter'], best_config['platform_level'], linestyle='-',
                color='blue', linewidth=2, label='Platform Level')
    axs[0].plot(best_config['iter'], best_config['mean_level'], linestyle='-',
                color='orange', linewidth=2, label='Mean Level')
    axs[0].axhline(y=level, color='red', linestyle='-', linewidth=3, alpha=0.7,
                   label=f'Target Level ({level})')

    # Add markers at change points
    change_points = df[is_new_min]
    for i, row in change_points.iterrows():
        axs[0].plot(row['iter'], row['platform_level'], 'bo', markersize=8)
        axs[0].plot(row['iter'], row['mean_level'], 'o', color='orange', markersize=8)

    axs[0].set_ylabel('Level')
    axs[0].set_title('Platform & Mean Level Changes Over Iterations (Best Configuration)')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(loc='best')

    # Plot 2: Spike Level
    axs[1].plot(best_config['iter'], best_config['spike_level'], linestyle='-',
                color='green', linewidth=2, label='Spike Level')
    axs[1].axhline(y=spike_level, color='darkgreen', linestyle='-', linewidth=3, alpha=0.7,
                   label=f'Target Spike Level ({spike_level})')

    # Add markers at change points
    for i, row in change_points.iterrows():
        axs[1].plot(row['iter'], row['spike_level'], 'go', markersize=8)

    axs[1].set_ylabel('Level')
    axs[1].set_title('Spike Level Changes Over Iterations (Best Configuration)')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend(loc='best')

    # Plot 3: Platform & Mean Length
    axs[2].plot(best_config['iter'], best_config['platform_length'], linestyle='-',
                color='blue', linewidth=2, label='Platform Length')
    axs[2].plot(best_config['iter'], best_config['mean_length'], linestyle='-',
                color='orange', linewidth=2, label='Mean Length')
    axs[2].axhline(y=length, color='red', linestyle='-', linewidth=3, alpha=0.7,
                   label=f'Target Length ({length})')

    # Add markers at change points
    for i, row in change_points.iterrows():
        axs[2].plot(row['iter'], row['platform_length'], 'bo', markersize=8)
        axs[2].plot(row['iter'], row['mean_length'], 'o', color='orange', markersize=8)

    axs[2].set_ylabel('Length')
    axs[2].set_title('Platform & Mean Length Changes Over Iterations (Best Configuration)')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend(loc='best')

    # Plot 4: Spike Probability
    axs[3].plot(best_config['iter'], best_config['spike_p'], linestyle='-',
                color='green', linewidth=2, label='Spike Probability')
    axs[3].axhline(y=spike_p, color='darkgreen', linestyle='-', linewidth=3, alpha=0.7,
                   label=f'Target Probability ({spike_p})')

    # Add markers at change points
    for i, row in change_points.iterrows():
        axs[3].plot(row['iter'], row['spike_p'], 'go', markersize=8)

    axs[3].set_xlabel('Iteration')
    axs[3].set_ylabel('Probability')
    axs[3].set_title('Spike Probability Changes Over Iterations (Best Configuration)')
    axs[3].grid(True, linestyle='--', alpha=0.7)
    axs[3].legend(loc='best')

    # Add an overall title
    plt.suptitle('Parameter Evolution Using Best Configuration at Each Iteration',
                 fontsize=16, y=0.995)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')

    return fig


def plot_running_best_only(input_filepath, output_filepath, type="wd", anomaly="platform", best_value=None,
                           baseline=None):
    df = pd.read_csv(input_filepath)
    plt.figure(figsize=(12, 10))

    column_name = "f1score" if type == "f1" else type

    if type == "wd":
        # Compute running minimum
        df['running_best'] = df[column_name].cummin()
        best_label = 'Running Min WD'

        # Global min
        global_best = df[column_name].min()
        global_idx = df[column_name].idxmin()
        global_iter = df.loc[global_idx, 'iter']
        global_label = 'Global Min WD'

        y_max = min(df[column_name].quantile(0.75) * 1.5, df[column_name].median() * 5)
        df['running_best_capped'] = df['running_best'].clip(upper=y_max)

        # Plot only running best
        plt.plot(df['iter'], df['running_best_capped'], marker='o', linestyle='--',
                 color='green', linewidth=2, label=best_label)
    else:
        # Compute running maximum
        df['running_best'] = df[column_name].cummax()
        best_label = 'Running Max F1'

        # Global max
        global_best = df[column_name].max()
        global_idx = df[column_name].idxmax()
        global_iter = df.loc[global_idx, 'iter']
        global_label = 'Global Max F1'

        plt.plot(df['iter'], df['running_best'], marker='o', linestyle='--',
                 color='green', linewidth=2, label=best_label)

    # Mark global best
    y_global = y_max if (type == "wd" and global_best > y_max) else global_best
    plt.scatter(global_iter, y_global, color='red', s=500, marker='*',
                zorder=5, label=f'{global_label} ({global_best:.4f})')

    # Horizontal line for provided best
    if best_value is not None:
        best_type = "Best WD" if type == "wd" else "Best F1"
        if type == "wd" and best_value > y_max:
            plt.text(0.02, 0.98, f'{best_type} = {best_value:.4f} (beyond scale)',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
        else:
            plt.axhline(y=best_value, color='red', linestyle='-', linewidth=2,
                        label=f'{best_type} = {best_value:.4f}')

    # Horizontal line for baseline
    if baseline is not None:
        baseline_type = "Baseline WD" if type == "wd" else "Baseline F1"
        if type == "wd" and baseline > y_max:
            plt.text(0.02, 0.92, f'{baseline_type} = {baseline:.4f} (beyond scale)',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
        else:
            plt.axhline(y=baseline, color='purple', linestyle='-', linewidth=2,
                        label=f'{baseline_type} = {baseline:.4f}')

    plt.xlabel("Iteration")
    plt.ylabel(type.upper())
    plt.title(f"Running Best {type.upper()} for {anomaly} anomalies")

    if type == "wd":
        plt.text(0.5, 0.01, f"Note: Y-axis capped at {y_max:.2f} to focus on smaller WD values",
                 ha='center', transform=plt.gcf().transFigure, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.1))

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.show()


if __name__ == "__main__":
    # plot_loss_curve(last=False)
    # plot_loss_curve(last=True)
    # plot_wd_f1score()
    _plot_wd_f1score()
    # plot_classifier_loss_curve('level_20')
    # plot_wd_f1score_combined()
    # plot_wd_f1score_spike()
    raise Exception()
    anomaly = "all"
    type = "wd"
    best_value = 0.09507475793361664
    baseline = 0.4806373715400696
    spike_level = 15
    spike_p = 0.03

    plot_running_best_only(
        input_filepath=f"logs/csv/six_anomalies/bayes_wd_f1score_{anomaly}.csv",
        output_filepath=f"logs/csv/six_anomalies/bayes_{type}_{anomaly}.png",
        anomaly=anomaly,
        type=type,
        best_value=best_value,
        baseline=baseline)
    # plot_level_length_step_function(
    #     input_filepath=f"logs/csv/hpo_three/bayes_wd_f1score_{anomaly}_{level}_{length}_{kappa}.csv",
    #     output_filepath=f"logs/graphs/hpo_three/bayes_{anomaly}_{level}_{length}_{kappa}_graph.png",
    #     level=level,
    #     length=length,
    #     spike_level = spike_level,
    #     spike_p = spike_p
    # )

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

# trail = 'second_anomaly'
trail = 'inject_spike'


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

    if trail == 'inject_spike':
        anomaly_types = ['platform', 'mean', 'spike']
    elif trail == 'second_anomaly':
        anomaly_types = ['platform', 'mean']
    else:
        raise Exception('Unsupported trail.')

    for anomaly_type in anomaly_types:
        level_wd = wd[anomaly_type]['level']
        level_f1 = f1score[anomaly_type]['level']
        length_wd = wd[anomaly_type]['length']
        length_f1 = f1score[anomaly_type]['length']

        def plot_heatmap(data, title, config_name):
            if config_name == 'level':
                configs = np.round(np.arange(-1.0, 1.1, 0.1), 1)
                x_config, y_config = np.meshgrid(configs, configs)
            else:  # config_name == 'length'
                configs = np.round(np.arange(0.20, 0.52, 0.02), 2)
                x_config, y_config = np.meshgrid(configs, configs)

            x_values = sorted(data.keys())
            y_values = sorted(data[x_values[0]].keys())
            values = np.zeros((len(y_values), len(x_values)))
            for i, x in enumerate(x_values):
                for j, y in enumerate(y_values):
                    values[j, i] = data[x][y]
            plt.figure(figsize=(8, 6))
            plt.pcolormesh(y_config, x_config, np.ma.masked_where(values == 0, values), cmap="viridis",
                           vmin=np.min(values), vmax=np.max(values))
            if title[-2:] == 'WD':
                for i in range(values.shape[1]):
                    column = values[:, i]
                    min_index = np.argmin(column)
                    plt.scatter(x_config[min_index, i], y_config[min_index, i], color='red', s=50, edgecolor='black',
                                label='Min Value')
            else:
                for i in range(values.shape[1]):
                    column = values[:, i]
                    max_index = np.argmax(column)
                    plt.scatter(x_config[max_index, i], y_config[max_index, i], color='red', s=50, edgecolor='black',
                                label='Max Value')
            plt.xticks(configs, rotation=90)
            plt.yticks(configs)
            plt.colorbar(label='Value')
            plt.xlabel(f'train_{config_name}')
            plt.ylabel(f'test_{config_name}')
            plt.title(title)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left')
            plt.savefig(f'logs/training/{trail}/{config_name}_{title}.pdf')
            plt.show()

        plot_heatmap(data=level_wd, title=f'{anomaly_type} WD', config_name='level')
        plot_heatmap(data=level_f1, title=f'{anomaly_type} F1-score', config_name='level')

        plot_heatmap(data=length_wd, title=f'{anomaly_type} WD', config_name='length')
        plot_heatmap(data=length_f1, title=f'{anomaly_type} F1-score', config_name='length')


#
# def plot_wd_f1score_combined():
#     with open(f"logs/training/{trail}/wd_f1score.txt", "r") as f:
#         lines = f.readlines()
#         wd = ast.literal_eval(lines[0][4:])
#         f1score = ast.literal_eval(lines[1][9:])
#
#     levels = np.round(np.arange(-1.0, 1.1, 0.1), 1)
#     lengths = np.round(np.arange(0.20, 0.52, 0.02), 2)
#     if trail == 'inject_spike':
#         anomaly_types = ['platform', 'mean', 'spike']
#     elif trail == 'second_anomaly':
#         anomaly_types = ['platform', 'mean']
#     else:
#         raise Exception('Unsupported trail.')
#     fixed_config = {'platform': {'level': 0.5, 'length': 0.3}, 'mean': {'level': 0.5, 'length': 0.3}}
#     configs = {'level': levels, 'length': lengths}
#     coordinate = list()
#     n = 0
#     if trail == 'inject_spike':
#         for anomaly_type in anomaly_types:
#             if anomaly_type == 'platform':
#                 for config in configs['level']:
#                     n += 1
#                     coordinate.append(
#                         f"({config}, {fixed_config['platform']['length']}, {fixed_config['mean']['level']}, "
#                         f"{fixed_config['mean']['length']})/ {n}")
#                 for config in configs['length']:
#                     n += 1
#                     coordinate.append(
#                         f"({fixed_config['platform']['level']}, {config}, {fixed_config['mean']['level']}, "
#                         f"{fixed_config['mean']['length']})/ {n}")
#             elif anomaly_type == 'mean':
#                 for config in configs['level']:
#                     n += 1
#                     coordinate.append(f"({fixed_config['platform']['level']}, {fixed_config['platform']['length']}, "
#                                       f"{config}, {fixed_config['mean']['length']})/ {n}")
#                 for config in configs['length']:
#                     n += 1
#                     coordinate.append(f"({fixed_config['platform']['level']}, {fixed_config['platform']['length']}, "
#                                       f"{fixed_config['mean']['level']}, {config})/ {n}")
#             else:
#                 for config in configs['level']:
#                     n += 1
#                     coordinate.append(f"({fixed_config['platform']['level']}, {fixed_config['platform']['length']}, "
#                                       f"{config}, {fixed_config['mean']['length']})/ {n}")
#                 for config in configs['length']:
#                     n += 1
#                     coordinate.append(f"({fixed_config['platform']['level']}, {fixed_config['platform']['length']}, "
#                                       f"{fixed_config['mean']['level']}, {config})/ {n}")
#
#     else:  # trail == 'second_anomaly'
#         for anomaly_type in anomaly_types:
#             if anomaly_type == 'platform':
#                 for config in configs['level']:
#                     n += 1
#                     coordinate.append(
#                         f"({config}, {fixed_config['platform']['length']}, {fixed_config['mean']['level']}, "
#                         f"{fixed_config['mean']['length']})/ {n}")
#                 for config in configs['length']:
#                     n += 1
#                     coordinate.append(
#                         f"({fixed_config['platform']['level']}, {config}, {fixed_config['mean']['level']}, "
#                         f"{fixed_config['mean']['length']})/ {n}")
#             else:
#                 for config in configs['level']:
#                     n += 1
#                     coordinate.append(f"({fixed_config['platform']['level']}, {fixed_config['platform']['length']}, "
#                                       f"{config}, {fixed_config['mean']['length']})/ {n}")
#                 for config in configs['length']:
#                     n += 1
#                     coordinate.append(f"({fixed_config['platform']['level']}, {fixed_config['platform']['length']}, "
#                                       f"{fixed_config['mean']['level']}, {config})/ {n}")
#
#     def plot_heatmap(data, title):
#         x = np.arange(len(coordinate))
#         y = np.arange(len(coordinate))
#
#         values = np.zeros((len(coordinate), len(coordinate)))
#         levels_num = len(levels)
#         lengths_num = len(lengths)
#
#         for i1, train_anomaly in enumerate(anomaly_types):
#             for train_config_name in data[train_anomaly].keys():
#                 for i2, train_config in enumerate(data[train_anomaly][train_config_name].keys()):
#                     for j1, valid_anomaly in enumerate(anomaly_types):
#                         for valid_config_name in data[train_anomaly][train_config_name][train_config][
#                             valid_anomaly].keys():
#                             for j2, valid_config in enumerate(data[train_anomaly][train_config_name][train_config][
#                                                                   valid_anomaly][valid_config_name].keys()):
#                                 i = i1 * (levels_num + lengths_num)
#                                 j = j1 * (levels_num + lengths_num)
#                                 if train_config_name == 'level':
#                                     i += i2
#                                 else:  # train_config_name == 'length'
#                                     i += levels_num + i2
#                                 if valid_config_name == 'level':
#                                     j += j2
#                                 else:  # valid_config_name == 'length'
#                                     j += levels_num + j2
#
#                                 values[i, j] = data[train_anomaly][train_config_name][train_config][valid_anomaly][
#                                     valid_config_name][valid_config]
#
#         plt.figure(figsize=(20, 16))
#         plt.pcolormesh(x, y, np.ma.masked_where(values == 0, values), cmap="viridis",
#                        vmin=np.min(values), vmax=np.max(values))
#         if title[-2:] == 'WD':
#             for i in range(values.shape[1]):
#                 column = values[i, :]
#                 min_index = np.argmin(column)
#                 plt.scatter(i, min_index, color='red', s=50, edgecolor='black', label='Min Value')
#         else:
#             for i in range(values.shape[1]):
#                 column = values[i, :]
#                 max_index = np.argmax(column)
#                 plt.scatter(i, max_index, color='red', s=50, edgecolor='black', label='Max Value')
#         plt.xticks(ticks=x, labels=coordinate, rotation=90)
#         plt.yticks(ticks=y, labels=coordinate)
#         plt.colorbar(label='Value')
#         plt.xlabel(f'train')
#         plt.ylabel(f'test')
#         plt.title(title)
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         plt.legend(by_label.values(), by_label.keys(), loc='upper left')
#         plt.savefig(f'logs/training/{trail}/{title}.pdf')
#         plt.show()
#
#         # import plotly.express as px
#         #
#         # import pandas as pd
#         # df = pd.DataFrame(values, index=coordinate, columns=coordinate)
#         #
#         # fig = px.imshow(df, text_auto=False, color_continuous_scale='Viridis')
#         # fig.show()
#
#     plot_heatmap(data=wd, title=f'WD')
#     plot_heatmap(data=f1score, title=f'F1-score')

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
        plt.figure(figsize=(20, 16))
        plt.pcolormesh(num, num, np.ma.masked_where(values == 0, values), cmap="viridis", vmin=np.min(values),
                       vmax=np.max(values))
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


def plot_wd_f1score_line(input_filepath, output_filepath, type="wd", anomaly="platform", level=0.5, length=0.3):
    df = pd.read_csv(input_filepath)
    plt.figure(figsize=(12, 10))

    # Map 'f1' to 'f1-score' if needed (to match CSV column name)
    column_name = "f1-score" if type == "f1" else type

    # Plot original data points
    plt.plot(df['iter'], df[column_name], marker='o', linestyle='-', color='blue',
             linewidth=2, label='Actual Values')

    # Calculate and plot the running best values
    if type == "wd":
        # Calculate running minimum WD for each iteration
        df['running_best'] = df[column_name].cummin()
        best_label = 'Running Min WD'

        # Highlight global minimum
        global_best = df[column_name].min()
        global_idx = df[column_name].idxmin()
        global_iter = df.loc[global_idx, 'iter']
        global_label = 'Global Min WD'

    elif type == "f1":
        # Calculate running maximum F1 for each iteration
        df['running_best'] = df[column_name].cummax()
        best_label = 'Running Max F1'

        # Highlight global maximum
        global_best = df[column_name].max()
        global_idx = df[column_name].idxmax()
        global_iter = df.loc[global_idx, 'iter']
        global_label = 'Global Max F1'

    # Plot running best as a separate line
    plt.plot(df['iter'], df['running_best'], marker='o', linestyle='--',
             color='green', linewidth=1.5, label=best_label)

    # Highlight global best with a star
    plt.scatter(global_iter, global_best, color='red', s=500, marker='*',
                zorder=5, label=global_label)

    plt.xlabel("Iteration")
    plt.ylabel(type)
    plt.title(f"Iteration vs {type} for {anomaly} anomaly ({level}, {length})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.show()


def plot_level_length_changes(input_filepath, output_filepath, anomaly="platform", level=0.5, length=0.3):
    # Read the CSV file
    df = pd.read_csv(input_filepath)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # Plot level changes in the top subplot
    ax1.plot(df['iter'], df['platform_level'], marker='o', linestyle='-',
             color='blue', linewidth=2, label='Platform Level')
    ax1.plot(df['iter'], df['mean_level'], marker='s', linestyle='-',
             color='orange', linewidth=2, label='Mean Level')

    # Add a horizontal line for the target level
    ax1.axhline(y=level, color='red', linestyle='--', alpha=0.7,
                label=f'Target Level ({level})')

    # Set up the top subplot
    ax1.set_ylabel('Level', fontsize=12)
    ax1.set_title(f'Level Changes Over Iterations for {anomaly.capitalize()} Anomaly',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best')

    # Plot length changes in the bottom subplot
    ax2.plot(df['iter'], df['platform_length'], marker='o', linestyle='-',
             color='blue', linewidth=2, label='Platform Length')
    ax2.plot(df['iter'], df['mean_length'], marker='s', linestyle='-',
             color='orange', linewidth=2, label='Mean Length')

    # Add a horizontal line for the target length
    ax2.axhline(y=length, color='red', linestyle='--', alpha=0.7,
                label=f'Target Length ({length})')

    # Set up the bottom subplot
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Length', fontsize=12)
    ax2.set_title(f'Length Changes Over Iterations for {anomaly.capitalize()} Anomaly',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best')

    # Calculate and mark the final distance to target values
    last_iter = df['iter'].max()
    last_idx = df['iter'].idxmax()

    last_platform_level = df.loc[last_idx, 'platform_level']
    last_platform_length = df.loc[last_idx, 'platform_length']

    # Annotate the final points
    ax1.annotate(f'Final: {last_platform_level:.4f}',
                 xy=(last_iter, last_platform_level),
                 xytext=(last_iter - 1, last_platform_level),
                 arrowprops=dict(arrowstyle='->'),
                 fontsize=10)

    ax2.annotate(f'Final: {last_platform_length:.4f}',
                 xy=(last_iter, last_platform_length),
                 xytext=(last_iter - 1, last_platform_length),
                 arrowprops=dict(arrowstyle='->'),
                 fontsize=10)

    # Add a subtitle with optimization information
    plt.figtext(0.5, 0.01,
                f"Target values: Level={level}, Length={length} | "
                f"Final platform values: Level={last_platform_level:.4f}, Length={last_platform_length:.4f}",
                ha='center', fontsize=12, bbox=dict(boxstyle='round', alpha=0.1))

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save and show the figure
    plt.savefig(output_filepath)
    plt.show()

    return fig


if __name__ == "__main__":
    # plot_loss_curve(last=False)
    # plot_loss_curve(last=True)
    # plot_wd_f1score()
    # plot_wd_f1score_combined()
    anomaly = "both"
    type = "f1"
    level = 0.5
    length = 0.3
    if anomaly == "spike":
        plot_wd_f1score_line(
            input_filepath=f"logs/training/inject_spike/bayes_{anomaly}_{level}_logs.csv",
            output_filepath=f"logs/training/inject_spike/bayes_{anomaly}_{level}_{type}.png",
            type=type)
    else:
        # plot_wd_f1score_line(
        #     input_filepath = f"logs/training/hpo_both/bayes_wd_f1score_{anomaly}_{level}_{length}_logs.csv",
        #     output_filepath = f"logs/training/hpo_both/bayes_{type}_{anomaly}_{level}_{length}.png",
        #     anomaly = anomaly,
        #     level = level,
        #     length = length,
        #     type = type)
        plot_level_length_changes(
            input_filepath=f"logs/training/hpo_both/bayes_wd_f1score_{anomaly}_{level}_{length}_logs.csv",
            output_filepath=f"logs/training/hpo_both/bayes_{type}_{anomaly}_{level}_{length}_graph.png",
            anomaly=anomaly,
            level=level,
            length=length
        )
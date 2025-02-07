import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

trail = 'fixed'
# trail = 'grid'
# trail = 'wo_first'
# trail = 'more_negative'
skip = 20


def plot_loss_curve():
    df = pd.read_csv(f'logs/training/{trail}/metrics.csv').dropna(subset=["val_loss"])
    epoch = df["epoch"]
    plt.figure(figsize=(6, 4))
    plt.plot(epoch, df["loss_global"], marker="s", markersize=10, markerfacecolor='none', linestyle="-",
             markeredgecolor="red", color="red", label=f"First Loss", markevery=skip)
    plt.plot(epoch, df["loss_local"], marker="^", markersize=10, markerfacecolor='none', linestyle="-",
             markeredgecolor="blue", color="blue", label=f"Second Loss", markevery=skip)
    plt.plot(epoch, df["loss_normal"], marker="v", markersize=10, markerfacecolor='none', linestyle="-",
             markeredgecolor="green", color="green", label=f"Third Loss", markevery=skip)
    plt.plot(epoch, df["val_loss"], marker="o", markersize=10, markerfacecolor='none', linestyle="-",
             markeredgecolor="black", color="black", label=f"Overall Loss", markevery=skip)
    plt.xticks(np.arange(0, len(epoch), skip))
    plt.title("Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'logs/training/{trail}/loss_curve.pdf')
    plt.close()


def plot_wd_f1score():
    with open(f"logs/training/{trail}/wd_f1score.txt", "r") as f:
        lines = f.readlines()
        wd = ast.literal_eval(lines[0][4:])
        f1score = ast.literal_eval(lines[1][9:])

    level_wd = wd['level']
    level_f1 = f1score['level']
    length_wd = wd['length']
    length_f1 = f1score['length']

    def plot_heatmap(data, title, config_name):
        if config_name == 'level':
            configs = np.round(np.arange(-1.0, 1.1, 0.1), 1)
            x_config, y_config = np.meshgrid(configs, configs)
        elif config_name == 'length':
            configs = np.round(np.arange(0.20, 0.52, 0.02), 2)
            x_config, y_config = np.meshgrid(configs, configs)
        else:
            raise Exception('Unsupported config')

        x_values = sorted(data.keys())
        y_values = sorted(data[x_values[0]].keys())
        values = np.zeros((len(y_values), len(x_values)))
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                if title == 'WD':
                    values[j, i] = -data[x][y]
                else:
                    values[j, i] = data[x][y]
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(y_config, x_config, np.ma.masked_where(values == 0, values), shading="auto", cmap="viridis",
                       vmin=np.min(values), vmax=np.max(values))
        if title == 'WD':
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
        plt.xlabel(config_name)
        plt.ylabel(config_name)
        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')
        plt.savefig(f'logs/training/{trail}/{config_name}_{title}.pdf')
        plt.show()

    plot_heatmap(data=level_wd, title='WD', config_name='level')
    plot_heatmap(data=level_f1, title='F1-score', config_name='level')

    plot_heatmap(data=length_wd, title='WD', config_name='length')
    plot_heatmap(data=length_f1, title='F1-score', config_name='length')


if __name__ == "__main__":
    plot_loss_curve()
    # plot_wd_f1score()

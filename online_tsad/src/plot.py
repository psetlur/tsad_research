import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

# trail = 'fixed'
# trail = 'grid'
# trail = 'more_epochs'
# trail = 'second_loss'
# trail = 'length_optimized'
# trail = 'more_negative'
# trail = 'warmup'
trail = 'second_anomaly'


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
        anomaly_types = ['platform', 'mean']
        for anomaly_type in anomaly_types:

            level_wd = wd[anomaly_type]['level']
            level_f1 = f1score[anomaly_type]['level']
            length_wd = wd[anomaly_type]['length']
            length_f1 = f1score[anomaly_type]['length']

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

def plot_combined_metrics():
    with open(f"logs/training/{trail}/wd_f1score.txt", "r") as f:
        lines = f.readlines()
        wd = ast.literal_eval(lines[0][4:])
        f1score = ast.literal_eval(lines[1][9:])
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Comparison of Level and Length Effects on Anomaly Detection', fontsize=16)
    
    axs[0].set_title('Best WD Scores')
    axs[1].set_title('Best F1 Scores')
    
    level_values = np.round(np.arange(-1.0, 1.1, 0.1), 1)
    length_values = np.round(np.arange(0.20, 0.52, 0.02), 2)
    
    styles = {
        ('platform', 'level'): {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'Platform (Level)'},
        ('platform', 'length'): {'color': 'blue', 'marker': 's', 'linestyle': '--', 'label': 'Platform (Length)'},
        ('mean', 'level'): {'color': 'red', 'marker': '^', 'linestyle': '-', 'label': 'Mean (Level)'},
        ('mean', 'length'): {'color': 'red', 'marker': 'D', 'linestyle': '--', 'label': 'Mean (Length)'}
    }
    
    # get best performance for each anomaly type and parameter type
    for anomaly_type in ['platform', 'mean']:
        for param_type in ['level', 'length']:
            if param_type == 'level':
                param_values = level_values
            else:
                param_values = length_values
            
            best_wd = []
            best_f1 = []
            
            # For each train parameter value
            for param in param_values:
                param_str = str(param)
                
                # Skip if this parameter isn't in the data
                if param_str not in wd[anomaly_type][param_type]:
                    best_wd.append(np.nan)  # Use np.nan for missing values
                    best_f1.append(np.nan)
                    continue
                
                # find min WD score
                min_wd = float('inf')
                for test_param in wd[anomaly_type][param_type][param_str]:
                    value = wd[anomaly_type][param_type][param_str][test_param]
                    if value < min_wd:
                        min_wd = value
                best_wd.append(min_wd if min_wd != float('inf') else np.nan)
                
                # find max F1 score
                max_f1 = 0
                for test_param in f1score[anomaly_type][param_type][param_str]:
                    value = f1score[anomaly_type][param_type][param_str][test_param]
                    if value > max_f1:
                        max_f1 = value
                best_f1.append(max_f1)
            
            # plot data with style
            style = styles[(anomaly_type, param_type)]
            axs[0].plot(param_values, best_wd, color=style['color'], marker=style['marker'], 
                       linestyle=style['linestyle'], label=style['label'])
            axs[1].plot(param_values, best_f1, color=style['color'], marker=style['marker'], 
                       linestyle=style['linestyle'], label=style['label'])
    
    for i in range(2):
        axs[i].set_xlabel('Parameter Value')
        axs[i].legend()
        axs[i].grid(True, linestyle='--', alpha=0.7)
    
    axs[0].set_ylabel('Wasserstein Distance (Lower is Better)')
    axs[1].set_ylabel('F1 Score (Higher is Better)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title
    plt.savefig(f'logs/training/{trail}/combined_metrics_comparison.pdf')
    plt.show()
    plt.close()


if __name__ == "__main__":
    # plot_loss_curve(last=False)
    # plot_loss_curve(last=True)
    # plot_wd_f1score()
    plot_combined_metrics()

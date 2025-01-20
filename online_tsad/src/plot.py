import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


def wd_f1score():
    wd = {}
    f1score = {}
    files = os.listdir('../logs/training')
    viridis_r = plt.get_cmap('viridis_r')
    new_colors_r = viridis_r(np.linspace(0, 1, 256))
    new_colors_r[-1] = [0, 0, 0, 1]
    new_cmap_r = ListedColormap(new_colors_r)

    # level
    x = np.round(np.arange(-1.0, 1.1, 0.1), 1)
    y = np.round(np.arange(-1.0, 1.1, 0.1), 1)
    X, Y = np.meshgrid(x, y)
    fixed_length = '0.30'
    for file in files:
        if f"length{fixed_length}" in file:
            fixed_level = float(file[5:-10])
            with open(f"training/{file}/wd_f1score_level.txt", "r") as f:
                lines = f.readlines()
                wd[fixed_level] = ast.literal_eval(lines[0][4:])
                f1score[fixed_level] = ast.literal_eval(lines[1][9:])
    WD = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            WD[i, j] = wd[xi][yj]
    mask = np.ma.masked_where(WD == 0, WD)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(Y, X, mask, shading="auto", cmap="viridis", vmin=np.min(WD[WD != 0]), vmax=np.max(WD))
    plt.gca().set_facecolor('white')
    plt.xticks(x, rotation=90)
    plt.yticks(y)
    plt.colorbar(label="Value")
    plt.title("WD Values")
    plt.xlabel("fixed_level")
    plt.ylabel("varying_level")
    plt.show()
    F1 = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            F1[i, j] = f1score[xi][yj]
    mask = np.ma.masked_where(F1 == 0, F1)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(Y, X, mask, shading="auto", cmap=new_cmap_r, vmin=np.min(F1[F1 != 0]), vmax=np.max(F1))
    plt.gca().set_facecolor('white')
    plt.xticks(x, rotation=90)
    plt.yticks(y)
    plt.colorbar(label="Value")
    plt.title("F1 Score Values")
    plt.xlabel("fixed_level")
    plt.ylabel("varying_level")
    plt.show()

    # length
    x = np.round(np.arange(0.20, 0.52, 0.02), 2)
    y = np.round(np.arange(0.20, 0.52, 0.02), 2)
    X, Y = np.meshgrid(x, y)
    fixed_level = 0.5
    for file in files:
        if f"level{fixed_level}" in file:
            fixed_length = file[14:]
            with open(f"training/{file}/wd_f1score_length.txt", "r") as f:
                lines = f.readlines()
                wd[fixed_length] = ast.literal_eval(lines[0][4:])
                f1score[fixed_length] = ast.literal_eval(lines[1][9:])
    WD = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            WD[i, j] = wd["{:.2f}".format(xi)]["{:.2f}".format(yj)]
    mask = np.ma.masked_where(WD == 0, WD)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(Y, X, mask, shading="auto", cmap="viridis", vmin=np.min(WD[WD != 0]), vmax=np.max(WD))
    plt.gca().set_facecolor('white')
    plt.xticks(x, rotation=90)
    plt.yticks(y)
    plt.colorbar(label="Value")
    plt.title("WD Values")
    plt.xlabel("fixed_length")
    plt.ylabel("varying_length")
    plt.show()
    F1 = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            F1[i, j] = f1score["{:.2f}".format(xi)]["{:.2f}".format(yj)]
    mask = np.ma.masked_where(F1 == 0, F1)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(Y, X, mask, shading="auto", cmap=new_cmap_r, vmin=np.min(F1[F1 != 0]), vmax=np.max(F1))
    plt.gca().set_facecolor('white')
    plt.xticks(x, rotation=90)
    plt.yticks(y)
    plt.colorbar(label="Value")
    plt.title("F1 Score Values")
    plt.xlabel("fixed_length")
    plt.ylabel("varying_length")
    plt.show()


def convergence(args):
    df = pd.read_csv(f'logs/training/level{args.fixed_level}length{"{:.2f}".format(args.fixed_length)}/metrics.csv')
    valid_data = df.dropna(subset=["val_loss"])
    valid_epochs = valid_data["epoch"]
    valid_val_losses = valid_data["val_loss"]
    plt.figure(figsize=(10, 6))
    plt.plot(valid_epochs, valid_val_losses, marker="o", linestyle="-", color="b", label="Validation Loss")
    plt.xticks(np.arange(0, len(valid_epochs), 10))
    plt.title("Validation Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig(f'logs/training/level{args.fixed_level}length{"{:.2f}".format(args.fixed_length)}/convergence.pdf')
    # plt.show()
    plt.close()

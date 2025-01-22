import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(f'training/fixed_grid/metrics.csv')
epochs = df["epoch"]
val_losses = df["val_loss"]
valid_data = df.dropna(subset=["val_loss"])
valid_epochs = valid_data["epoch"]
valid_val_losses = valid_data["val_loss"]
plt.figure(figsize=(10, 6))
plt.plot(valid_epochs, valid_val_losses, marker="o", linestyle="-", color="b", label="Validation Loss", markevery=10)
plt.xticks(np.arange(0, len(valid_epochs), 10))
plt.title("Validation Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig(f'training/fixed_grid/convergence.pdf')
plt.close()

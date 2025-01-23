import numpy as np
import math

import torch
from torch import nn
import geomloss

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import logging
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from matplotlib.colors import Normalize

cmap = cm.get_cmap('Reds')
norm = Normalize(vmin=-1, vmax=1)
levels = np.round(np.arange(-1.0, 1.1, 0.1), 1)

for i in levels:
    print(cmap(norm(i)))

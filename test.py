import torch
import pandas as pd

from torch.utils.data import random_split
from torch.utils.data import Subset

from dataset import Dataset

path = ["airFoil2D_SST_58.831_-3.563_2.815_4.916_10.078"]

data, coef_norm = Dataset(path, norm=True, sample=None)

df_input = pd.DataFrame(data[0].x, columns=["x", "y", "u_inlet", "v_inlet", "sdf", "x_normal", "y_normal"])
df_target = pd.DataFrame(data[0].y, columns=["u", "v", "p", "nut"])

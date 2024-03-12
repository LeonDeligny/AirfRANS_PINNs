import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import torch
import os
import logging

from datetime import datetime
from scipy.interpolate import griddata
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from plot import figsize, newfig, savefig, plot_solution, axisEqual3D, plot_predictions_vs_test, pgf_with_latex
from dataset import Dataset
from model import PhysicsInformedNN

# Set up Python logging
logging.basicConfig(level=logging.ERROR)

# Create a log directory with a timestamp to keep different runs separate
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)

torch.manual_seed(1234)
        
if __name__ == "__main__":
    mpl.rcParams.update(pgf_with_latex)

    # NN Configuration
    layers = [8, 128, 128, 128, 128, 128, 4]

    # Load Data (change path if needed)
    path = ["airFoil2D_SST_58.831_-3.563_2.815_4.916_10.078"]

    data, coef_norm = Dataset(path, norm=True, sample=None)

    df_input = pd.DataFrame(data[0].x, columns=["x", "y", "u_inlet", "v_inlet", "sdf", "x_normal", "y_normal"])
    df_target = pd.DataFrame(data[0].y, columns=["u", "v", "p", "nut"])
    df_surf = pd.DataFrame(data[0].surf, columns=["surf_bool"])

    XY = df_input[['x', 'y']].values
    XY_NORMAL = df_input[["x_normal", "y_normal"]].values
    SDF = df_input[['sdf']].values
    UV = df_target[['u', 'v']].values
    P = df_target[['p']].values
    NUT = df_target[['nut']].values
    SURF = df_surf[['surf_bool']].values

    # Constants
    gamma_values = np.array([2.815, 4.916, 10.078]).reshape(-1, 1)
    N = XY.shape[0]

    zero_normal_indices = np.where(SURF == True)[0]
    remaining_indices = np.setdiff1d(np.arange(N), zero_normal_indices)
    N_train = int(0.3 * remaining_indices.shape[0])

    id_train = np.random.choice(remaining_indices, N_train, replace=False)
    id_train = np.union1d(id_train, zero_normal_indices)
    train_length = len(id_train)
    id_test = np.setdiff1d(np.arange(N), id_train)

    def split_data(data):
        return data[id_train, :], data[id_test, :]

    x_train, x_test = split_data(XY[:, 0].reshape(-1, 1))
    y_train, y_test = split_data(XY[:, 1].reshape(-1, 1))
    x_normal_train, x_normal_test = split_data(XY_NORMAL[:, 0].reshape(-1, 1))
    y_normal_train, y_normal_test = split_data(XY_NORMAL[:, 1].reshape(-1, 1))
    u_train, u_test = split_data(UV[:, 0].reshape(-1, 1))
    v_train, v_test = split_data(UV[:, 1].reshape(-1, 1))
    nut_train, nut_test = split_data(NUT)
    p_train, p_test = split_data(P)
    sdf_train, sdf_test = split_data(SDF)

    gamma_1_train, gamma_1_test = np.full((train_length, 1), gamma_values[0]), np.full((N - train_length, 1), gamma_values[0])
    gamma_2_train, gamma_2_test = np.full((train_length, 1), gamma_values[1]), np.full((N - train_length, 1), gamma_values[1])
    gamma_3_train, gamma_3_test = np.full((train_length, 1), gamma_values[2]), np.full((N - train_length, 1), gamma_values[2])

    # Train the model
    model = PhysicsInformedNN(layers, coef_norm, 
                              u_train, v_train, p_train, nut_train, 
                              x_train, y_train, x_normal_train, y_normal_train, 
                              sdf_train, gamma_1_train, gamma_2_train, gamma_3_train
                              )
    model.train(12)

    # Prediction
    u_pred, v_pred, p_pred, nut_pred = model.predict(x_test, y_test, x_normal_test, y_normal_test, 
                                           sdf_test, gamma_1_test, gamma_2_test, gamma_3_test
                                          )

    # Plotting
    plot_predictions_vs_test(x_test, y_test, u_pred, u_test, 'u')
    plot_predictions_vs_test(x_test, y_test, v_pred, v_test, 'v')
    plot_predictions_vs_test(x_test, y_test, p_pred, p_test, 'p')
    plot_predictions_vs_test(x_test, y_test, nut_pred, nut_test, 'nut')

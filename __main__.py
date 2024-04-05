import numpy as np, pandas as pd
import torch, os, logging, optuna

from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from preprocessing import load_dataset, normalize
from plot import plot_predictions_vs_test, plot_test
from PIMNN_Phy_Bc import PIMNN_Phy_Bc

# Set up Python logging
logging.basicConfig(level=logging.ERROR)

# Create a log directory with a timestamp to keep different runs separate
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)

pd.set_option('display.precision', 20)

torch.manual_seed(1234)
torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    # Load Data (change path if needed)
    path = ["airFoil2D_SST_31.677_-0.742_2.926_5.236_5.651"]
    train_data, bc_data, box_data = load_dataset(path, 
                                       n_random_sampling = 100
                                       )

    Uinf, alpha, gamma_1, gamma_2, gamma_3 = float(path[0].split('_')[2]), float(path[0].split('_')[3])*np.pi/180, float(path[0].split('_')[4]), float(path[0].split('_')[5]), float(path[0].split('_')[6])
    print(f"Uinf: {Uinf}, alpha: {alpha}")
    
    u_inlet, v_inlet = np.cos(alpha)*Uinf, np.sin(alpha)*Uinf
    
    df_train_input = pd.DataFrame(train_data[0].x_train, columns=["x", "y", "sdf"])
    df_train_target = pd.DataFrame(train_data[0].y_train, columns=["u", "v", "p", "nut"])
    df_train = pd.concat([df_train_input, df_train_target], axis=1) 

    df_bc_input = pd.DataFrame(bc_data[0].x_bc, columns=["x", "y", "sdf"])
    df_bc_target = pd.DataFrame(bc_data[0].y_bc, columns=["u", "v", "p", "nut"])
    df_bc = pd.concat([df_bc_input, df_bc_target], axis=1) 

    df_combined = pd.concat([df_train, df_bc], axis=0)
    df_combined_normalized, mean_variance_dict = normalize(df_combined) # normalize(df_combined)

    l = len(df_train)

    df_train_normalized = df_combined_normalized.iloc[:l, :]
    df_bc_normalized = df_combined_normalized.iloc[l:, :]

    df_box_input = pd.DataFrame(box_data[0].x_box, columns=["x", "y", "sdf"])
    df_box_target = pd.DataFrame(box_data[0].y_box, columns=["u", "v", "p", "nut"])
    df_box = pd.concat([df_box_input, df_box_target], axis=1)

    l_prime = len(df_box)

    df_box_normalized = df_combined_normalized.iloc[:l_prime, :]
    
    print("Dataset loaded.")
    print(mean_variance_dict)
        # Train the model
    fourier_scale = 1
    fourier_mapdim = 64

    model = PIMNN_Phy_Bc(df_train_normalized, df_bc_normalized, mean_variance_dict, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3, fourier_scale, fourier_mapdim)
    model.pre_train(51)
    print(f"Finished pre-Training.")
    model.train(501)
    print(f"Finished Training.")
    # Prediction u_pred, v_pred, p_pred, nut_pred
    u_pred, v_pred, p_pred, nut_pred = model.predict(df_box_normalized, mean_variance_dict, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3)

    # Plotting
    plot_predictions_vs_test(df_box_normalized['x'].astype(float).values.flatten(), df_box_normalized['y'].astype(float).values.flatten(), u_pred, df_box_normalized['u'], 'u', 'PINN_u_Supervised')
    plot_predictions_vs_test(df_box_normalized['x'].astype(float).values.flatten(), df_box_normalized['y'].astype(float).values.flatten(), v_pred, df_box_normalized['v'], 'v', 'PINN_v_Supervised')
    plot_predictions_vs_test(df_box_normalized['x'].astype(float).values.flatten(), df_box_normalized['y'].astype(float).values.flatten(), p_pred, df_box_normalized['p'], 'p', 'PINN_p_Supervised')
    plot_predictions_vs_test(df_box_normalized['x'].astype(float).values.flatten(), df_box_normalized['y'].astype(float).values.flatten(), nut_pred, df_box_normalized['nut'], 'nut', 'PINN_nut_Supervised')
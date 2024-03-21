import numpy as np, pandas as pd, pyvista as pv, os.path as osp
import torch, os, logging

from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from plot import plot_predictions_vs_test, plot_test
from PINN_Phy_Bc import PINN_Phy_Bc
from PINN_Supervised import PINN_Supervised
from PIMNN import PIMNN
from PINN_p_Supervised import PINN_p_Supervised
from PINN_nut_Supervised import PINN_nut_Supervised


# Set up Python logging
logging.basicConfig(level=logging.ERROR)

# Create a log directory with a timestamp to keep different runs separate
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)

pd.set_option('display.precision', 20)

torch.manual_seed(1234)

def compute_minimum_distances(freestream, aerofoil):
    if hasattr(freestream, 'points'):
        freestream_points = freestream.points[:, :2]
    else:
        freestream_points = freestream

    distances = np.zeros(freestream_points.shape[0])
    for i, point_a in enumerate(freestream_points):
        min_distance = np.min(np.sqrt(np.sum((aerofoil.points[:, :2] - point_a) ** 2, axis=1)))
        distances[i] = min_distance
    return distances


def reorganize(in_order_points, out_order_points, quantity_to_reordered):
    n = out_order_points.shape[0]
    idx = np.zeros(n)
    for i in range(n):
        cond = (out_order_points[i] == in_order_points)
        cond = cond[:, 0]*cond[:, 1]        
        idx[i] = np.argwhere(cond)[0][0]
    idx = idx.astype('int')

    assert (in_order_points[idx] == out_order_points).all()

    return quantity_to_reordered[idx]


def sample_points(n):
    points = []
    while len(points) < n:
        x = np.random.uniform(-200, 200) 
        y = np.random.uniform(-200, 200)

        # Ensure the point is inside the semi-circle and not in the excluded rectangle
        if (((x**2 + y**2 <= 200**2) and (x <= 0)) or x > 0) and not (-2 <= x <= 4 and -1.5 <= y <= 1.5):
            points.append((x, y))
    return points


def load_dataset(path, n_random_sampling=0):
    train_dataset = []
    bc_dataset = []
    box_dataset = []
    for k, s in enumerate(tqdm(path)):
        # Get the 3D mesh, add the signed distance function and slice it to return in 2D
        internal = pv.read(osp.join('Dataset', s, s + '_internal.vtu'))

        internal = internal.compute_cell_sizes(length = False, volume = False)
        aerofoil = pv.read(osp.join('Dataset', s, s + '_aerofoil.vtp'))
        freestream = pv.read(osp.join('Dataset', s, s + '_freestream.vtp'))

        # u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(internal.point_data['U'][:, :1])
        geom = -internal.point_data['implicit_distance'][:, None] # Signed distance function
        # normal = np.zeros_like(u)

        # surf_bool = (internal.point_data['U'][:, 0] == 0)
        # normal[surf_bool] = reorganize( aerofoil.points[:, :2], internal.points[surf_bool, :2], -aerofoil.point_data['Normals'][:, :2]) # no "Normal" feature in internal dataset

        internal_attr = np.concatenate([   
                                            internal.points[:, :2],
                                            geom,
                                            # normal, 
                                            internal.point_data['U'][:, :2], 
                                            internal.point_data['p'][:, None], 
                                            internal.point_data['nut'][:, None],

                                        ], axis = -1)
        
        if n_random_sampling !=0 :
            points_sampled = np.array(sample_points(n_random_sampling))
            geom_sampled = compute_minimum_distances(points_sampled, aerofoil)[:, np.newaxis] 
            sampled_attr = np.concatenate([
                                            points_sampled,
                                            geom_sampled,

                                            ], axis = -1)
            
            columns_to_add = internal_attr.shape[1] - sampled_attr.shape[1]
            if columns_to_add > 0:
                nan_columns = np.full((sampled_attr.shape[0], columns_to_add), np.nan)
                sampled_attr = np.concatenate([sampled_attr, nan_columns], axis=1)

            internal_attr_sample = np.concatenate([internal_attr, sampled_attr], axis = 0)
        
        else:
            internal_attr_sample = internal_attr

        # freestream_normals = np.zeros((freestream.points.shape[0], 2)) 
        freestream_geom = compute_minimum_distances(freestream, aerofoil)
        freestream_geom = freestream_geom[:, np.newaxis]
        # freestream_u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(freestream.point_data['U'][:, :1])

        freestream_attr = np.concatenate([   
                                            freestream.points[:, :2],
                                            freestream_geom, 
                                            # freestream_normals,
                                            freestream.point_data['U'][:, :2], 
                                            freestream.point_data['p'][:, None], 
                                            freestream.point_data['nut'][:, None],
                                        
                                        ], axis = -1)
    
        
        init_train = np.concatenate([internal_attr_sample[:, :3]], axis = 1) # pos, inlet_vel, geom
        init_bc = np.concatenate([freestream_attr[:, :3]], axis = 1) # pos, inlet_vel, geom
        target_train = internal_attr_sample[:, 3:] # u, p, nut
        target_bc = freestream_attr[:, 3:]
        # surf_bool = (attr[:,4:5] == 0)
        init_box = np.concatenate([internal_attr[:, :3]], axis = 1) # pos, inlet_vel, geom
        target_box = internal_attr[:, 3:] # u, p, nut

        # Put everything in tensor
        x_train = torch.tensor(init_train, dtype = torch.float64)
        y_train = torch.tensor(target_train, dtype = torch.float64)
        x_bc = torch.tensor(init_bc, dtype = torch.float64)
        y_bc = torch.tensor(target_bc, dtype = torch.float64)
        x_box = torch.tensor(init_box, dtype = torch.float64)
        y_box = torch.tensor(target_box, dtype = torch.float64)

        # surf_bool = torch.tensor(surf_bool, dtype = torch.bool)£
        train_data = Data(x_train = x_train, y_train = y_train)
        bc_data = Data(x_bc = x_bc, y_bc = y_bc)
        box_data = Data(x_box = x_box, y_box = y_box)

        train_dataset.append(train_data)
        bc_dataset.append(bc_data)
        box_dataset.append(box_data)

    return train_dataset, bc_dataset, box_dataset


def normalize(df):
    scaler = StandardScaler()
    df[:] = scaler.fit_transform(df)
    mean_variance_dict = {column: {"mean": scaler.mean_[i], "var": scaler.var_[i]} for i, column in enumerate(df.columns)}
    return df, mean_variance_dict


if __name__ == "__main__":

    # Load Data (change path if needed)
    path = ["airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32"]
    train_data, bc_data, box_data = load_dataset(path, 
                                       n_random_sampling = 600000
                                       )

    Uinf, alpha, gamma_1, gamma_2, gamma_3 = float(path[0].split('_')[2]), float(path[0].split('_')[3])*np.pi/180, float(path[0].split('_')[4]), float(path[0].split('_')[5]), float(path[0].split('_')[6])
    print(f"Uinf: {Uinf}, alpha: {alpha}")
    
    u_inlet, v_inlet = np.cos(alpha)*Uinf, np.sin(alpha)*Uinf
    
    df_train_input = pd.DataFrame(train_data[0].x_train, columns=["x", "y", "sdf"])
    df_train_target = pd.DataFrame(train_data[0].y_train, columns=["u", "v", "p", "nut"])
    df_train = pd.concat([df_train_input, df_train_target], axis=1) 

    df_train, mean_variance_dict = normalize(df_train)

    df_bc_input = pd.DataFrame(bc_data[0].x_bc, columns=["x", "y", "sdf"])
    df_bc_target = pd.DataFrame(bc_data[0].y_bc, columns=["u", "v", "p", "nut"])
    df_bc = pd.concat([df_bc_input, df_bc_target], axis=1) 

    df_box_input = pd.DataFrame(box_data[0].x_box, columns=["x", "y", "sdf"])
    df_box_target = pd.DataFrame(box_data[0].y_box, columns=["u", "v", "p", "nut"])
    df_box = pd.concat([df_box_input, df_box_target], axis=1) 

    # NN Configuration
    layers = [8, 128, 128, 128, 128, 4]

    # Train the model
    model = PINN_Phy_Bc(layers, df_train, df_bc, mean_variance_dict, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3, fourier_scale=1)
    model.train(2000)

    # Prediction
    u_pred, v_pred, p_pred, nut_pred = model.predict(df_box, df_bc, mean_variance_dict, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3)

    # Plotting
    plot_predictions_vs_test(df_box['x'].astype(float).values.flatten(), df_box['y'].astype(float).values.flatten(), u_pred, df_box['u'], 'u', layers)
    plot_predictions_vs_test(df_box['x'].astype(float).values.flatten(), df_box['y'].astype(float).values.flatten(), v_pred, df_box['v'], 'v', layers)
    plot_predictions_vs_test(df_box['x'].astype(float).values.flatten(), df_box['y'].astype(float).values.flatten(), p_pred, df_box['p'], 'p', layers)
    plot_predictions_vs_test(df_box['x'].astype(float).values.flatten(), df_box['y'].astype(float).values.flatten(), nut_pred, df_box['nut'], 'nut', layers)

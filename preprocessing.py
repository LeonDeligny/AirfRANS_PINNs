import torch

import numpy as np, pyvista as pv, os.path as osp

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch_geometric.data import Data


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
    freestream_dataset = []
    aerofoil_dataset = []
    box_dataset = [] 
    for k, s in enumerate(tqdm(path)): 
        # Get the 3D mesh, add the signed distance function and slice it to return in 2D.
        
        # internal = pv.read('/notebooks/Dataset/' + s + '/' + s + '_internal.vtu')
        internal = pv.read(osp.join('Dataset', s, s + '_internal.vtu'))

        internal = internal.compute_cell_sizes(length = False, volume = False)
        # aerofoil = pv.read('/notebooks/Dataset/' + s + '/' + s + '_aerofoil.vtp')
        aerofoil = pv.read(osp.join('Dataset', s, s + '_aerofoil.vtp'))

        # freestream = pv.read('/notebooks/Dataset/' + s + '/' + s + '_freestream.vtp')
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
        
        # aerofoil are points where sdf = 0.
        aerofoil_attr = internal_attr[internal_attr[:, 2] == 0.0]
        
        # the rest consists of the internal space.
        internal_attr = internal_attr[internal_attr[:, 2] != 0.0]
        
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
                        
        init_train = np.concatenate([freestream_attr[:, :3], aerofoil_attr[:, :3], internal_attr_sample[:, :3]], axis = 0) 
   
        target_train = np.concatenate([freestream_attr[:, 3:], aerofoil_attr[:, 3:], internal_attr_sample[:, 3:]], axis = 0)

        # Put everything in tensor
        x_train = torch.tensor(init_train, dtype = torch.float64)
        y_train = torch.tensor(target_train, dtype = torch.float64)

        # surf_bool = torch.tensor(surf_bool, dtype = torch.bool)
        train_data = Data(x_train = x_train, y_train = y_train)

        train_dataset.append(train_data)

        len_list = [len(freestream_attr), len(aerofoil_attr), len(internal_attr), len(internal_attr_sample)]
        
    return train_dataset, len_list


def normalize(df):
    scaler = StandardScaler()
    df_copy = df.copy()
    df_copy[:] = scaler.fit_transform(df_copy)
    mean_variance_dict = {column: {"mean": scaler.mean_[i], "var": scaler.var_[i]} for i, column in enumerate(df_copy.columns)}
    return df_copy, mean_variance_dict
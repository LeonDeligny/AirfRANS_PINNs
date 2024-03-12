import numpy as np
import pyvista as pv
import os.path as osp
import torch

from torch_geometric.data import Data
from tqdm import tqdm

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

def cell_sampling_2d(cell_points, cell_attr = None):
    '''
    Sample points in a two dimensional cell via parallelogram sampling and triangle interpolation via barycentric coordinates. The vertices have to be ordered in a certain way.

    Args:
        cell_points (array): Vertices of the 2 dimensional cells. Shape (N, 4, 2) for N cells with 2 dimensional 4 vertices.
        cell_attr (array, optional): Features of the vertices of the 2 dimensional cells. Shape (N, 4, k) for N cells with 4 edges and k features. 
            If given shape (N, 4) it will resize it automatically in a (N, 4, 1) array. Default: ``None``
    '''
    # Sampling via triangulation of the cell and parallelogram sampling.
    v0, v1 = cell_points[:, 1] - cell_points[:, 0], cell_points[:, 3] - cell_points[:, 0]
    v2, v3 = cell_points[:, 3] - cell_points[:, 2], cell_points[:, 1] - cell_points[:, 2]  

    # Arrays of triangle surface a0 described by v0, v1.
    a0, a1 = np.abs(np.linalg.det(np.hstack([v0[:, :2], v1[:, :2]]).reshape(-1, 2, 2))), np.abs(np.linalg.det(np.hstack([v2[:, :2], v3[:, :2]]).reshape(-1, 2, 2)))
    
    # Array of probabilities of sampling a triangle in the parallelogram.
    p = a0/(a0 + a1)

    # Binomial distribution, 1 number of trials, p probability of success, result is binary (either 0 or 1).
    index_triangle = np.random.binomial(1, p)[:, None]

    # Array of size (len(p), 2) where the columns consist of random numbers between 0 and 1.
    u = np.random.uniform(size = (len(p), 2))

    # Array of sampled points.
    sampled_point = index_triangle*(u[:, 0:1]*v0 + u[:, 1:2]*v1) + (1 - index_triangle)*(u[:, 0:1]*v2 + u[:, 1:2]*v3)

    # Arry of mirrored sampled points.
    sampled_point_mirror = index_triangle*((1 - u[:, 0:1])*v0 + (1 - u[:, 1:2])*v1) + (1 - index_triangle)*((1 - u[:, 0:1])*v2 + (1 - u[:, 1:2])*v3)
    
    # Cases when points fall outside intended triangle.
    reflex = (u.sum(axis = 1) > 1)

    # Corrects these points with their counterparts. All sample points lie within the triangle.
    sampled_point[reflex] = sampled_point_mirror[reflex]

    # Interpolation on a triangle via barycentric coordinates
    if cell_attr is not None:
        # Vertice for Barycentric coordinates. 
        t0, t1, t2 = np.zeros_like(v0), index_triangle*v0 + (1 - index_triangle)*v2, index_triangle*v1 + (1 - index_triangle)*v3

        # Weights.
        w = (t1[:, 1] - t2[:, 1])*(t0[:, 0] - t2[:, 0]) + (t2[:, 0] - t1[:, 0])*(t0[:, 1] - t2[:, 1])
        w0 = (t1[:, 1] - t2[:, 1])*(sampled_point[:, 0] - t2[:, 0]) + (t2[:, 0] - t1[:, 0])*(sampled_point[:, 1] - t2[:, 1])
        w1 = (t2[:, 1] - t0[:, 1])*(sampled_point[:, 0] - t2[:, 0]) + (t0[:, 0] - t2[:, 0])*(sampled_point[:, 1] - t2[:, 1])
        w0, w1 = w0/w, w1/w
        w2 = 1 - w0 - w1
        
        if len(cell_attr.shape) == 2:
            cell_attr = cell_attr[:, :, None]
        attr0 = index_triangle*cell_attr[:, 0] + (1 - index_triangle)*cell_attr[:, 2]
        attr1 = index_triangle*cell_attr[:, 1] + (1 - index_triangle)*cell_attr[:, 1]
        attr2 = index_triangle*cell_attr[:, 3] + (1 - index_triangle)*cell_attr[:, 3]

        # Interpolated attributes (weighted average).
        sampled_attr = w0[:, None]*attr0 + w1[:, None]*attr1 + w2[:, None]*attr2

    sampled_point += index_triangle*cell_points[:, 0] + (1 - index_triangle)*cell_points[:, 2]    

    # Returns sampled points (and sampled attributes).
    return np.hstack([sampled_point[:, :2], sampled_attr]) if cell_attr is not None else sampled_point[:, :2]

def cell_sampling_1d(line_points, line_attr = None):
    '''
    Sample points in a one dimensional cell via linear sampling and interpolation.

    Args:
        line_points (array): Edges of the 1 dimensional cells. Shape (N, 2) for N cells with 2 edges.
        line_attr (array, optional): Features of the edges of the 1 dimensional cells. Shape (N, 2, k) for N cells with 2 edges and k features.
            If given shape (N, 2) it will resize it automatically in a (N, 2, 1) array. Default: ``None``
    '''
    # Linear/Uniform sampling.
    u = np.random.uniform(size = (len(line_points), 1))

    # Sample points on the segment.
    sampled_point = u*line_points[:, 0] + (1 - u)*line_points[:, 1]

    # Linear interpolation
    if line_attr is not None:   
        if len(line_attr.shape) == 2:
            line_attr = line_attr[:, :, None]
        sampled_attr = u*line_attr[:, 0] + (1 - u)*line_attr[:, 1]

    return np.hstack([sampled_point[:, :2], sampled_attr]) if line_attr is not None else sampled_point[:, :2]

def Dataset(set, norm=False, coef_norm=None, crop=None, sample=None, n_boot=int(5e5), surf_ratio=.1):
    '''
    Create a list of simulation to input in a PyTorch Geometric DataLoader. Simulation are transformed by keeping vertices of the CFD mesh or 
    by sampling (uniformly or via the mesh density) points in the simulation cells.

    Args:
        set (list): List of geometry names to include in the dataset.
        norm (bool, optional): If norm is set to ``True``, the mean and the standard deviation of the dataset will be computed and returned. 
            Moreover, the dataset will be normalized by these quantities. Ignored when ``coef_norm`` is not None. Default: ``False``
        coef_norm (tuple, optional): This has to be a tuple of the form (mean input, std input, mean output, std ouput) if not None. 
            The dataset generated will be normalized by those quantites. Default: ``None``
        crop (list, optional): List of the vertices of the rectangular [xmin, xmax, ymin, ymax] box to crop simulations. Default: ``None``
        sample (string, optional): Type of sampling. If ``None``, no sampling strategy is applied and the nodes of the CFD mesh are returned.
            If ``uniform`` or ``mesh`` is chosen, uniform or mesh density sampling is applied on the domain. Default: ``None``
        n_boot (int, optional): Used only if sample is not None, gives the size of the sampling for each simulation. Defaul: ``int(5e5)``
        surf_ratio (float, optional): Used only if sample is not None, gives the ratio of point over the airfoil to sample with respect to point
            in the volume. Default: ``0.1``
    '''
    if norm and coef_norm is not None:
        raise ValueError('If coef_norm is not None and norm is True, the normalization will be done via coef_norm')

    dataset = []

    mean_in, std_in, mean_out, std_out = None, None, None, None

    for k, s in enumerate(tqdm(set)):
        # Get the 3D mesh, add the signed distance function and slice it to return in 2D
        internal = pv.read(osp.join('Dataset', s, s + '_internal.vtu'))
        aerofoil = pv.read(osp.join('Dataset', s, s + '_aerofoil.vtp'))
        freestream = pv.read(osp.join('Dataset', s, s + '_freestream.vtp'))
        internal = internal.compute_cell_sizes(length = False, volume = False)

        internal_num_arrays = internal.point_data.GetNumberOfArrays()
        aerofoil_num_arrays = aerofoil.point_data.GetNumberOfArrays()
        freestream_num_arrays = freestream.point_data.GetNumberOfArrays()

        '''
        # Iterate through all arrays in vtkPointData
        for i in range(internal_num_arrays):
            array_name = internal.point_data.GetArrayName(i)
            print(f"Internal Array name: {array_name}")

        for i in range(aerofoil_num_arrays):
            array_name = aerofoil.point_data.GetArrayName(i)
            print(f"Aerofoil Array name: {array_name}")

        for i in range(freestream_num_arrays):
            array_name = freestream.point_data.GetArrayName(i)
            print(f"Freestream Array name: {array_name}")
        '''

        # Cropping if needed, crinkle is True.
        if crop is not None:
            bounds = (crop[0], crop[1], crop[2], crop[3], 0, 1)
            internal = internal.clip_box(bounds = bounds, invert = False, crinkle = True)

        # If sampling strategy is chosen, it will sample points in the cells of the simulation instead of directly taking the nodes of the mesh.
        if sample is not None:
            # Sample on a new point cloud
            if sample == 'uniform': # Uniform sampling strategy, sampling depends on Area & Length of cells
                p = internal.cell_data['Area']/internal.cell_data['Area'].sum() # probability of each cell being selected
                sampled_cell_indices = np.random.choice(internal.n_cells, size = n_boot, p = p) # randomly selects n_boot cells from the internal dataset
                surf_p = aerofoil.cell_data['Length']/aerofoil.cell_data['Length'].sum() # probability of each length being selected
                sampled_line_indices = np.random.choice(aerofoil.n_cells, size = int(n_boot*surf_ratio), p = surf_p) # randomly selects int(n_boot*surf_ratio) cells from the aerofoil dataset
            elif sample == 'mesh': # Sample via mesh density, sampling is random
                sampled_cell_indices = np.random.choice(internal.n_cells, size = n_boot)
                sampled_line_indices = np.random.choice(aerofoil.n_cells, size = int(n_boot*surf_ratio))

            cell_dict = internal.cells.reshape(-1, 5)[sampled_cell_indices, 1:] # [N]x5
            cell_points = internal.points[cell_dict]            
            line_dict = aerofoil.lines.reshape(-1, 3)[sampled_line_indices, 1:]
            line_points = aerofoil.points[line_dict]

            # Geometry information
            geom = -internal.point_data['implicit_distance'][cell_dict, None] # Signed distance function
            Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180
            u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(internal.point_data['U'][cell_dict, :1])
            normal = np.zeros_like(u)

            surf_geom = np.zeros_like(aerofoil.point_data['U'][line_dict, :1])
            surf_u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(aerofoil.point_data['U'][line_dict, :1])
            surf_normal = -aerofoil.point_data['Normals'][line_dict, :2]
            
            attr = np.concatenate([ u, # inlet_velocity
                                    geom, # -sdf
                                    normal, # normal vector (0, 0)
                                    internal.point_data['U'][cell_dict, :2], # velocity, keep only the first two dimensions
                                    internal.point_data['p'][cell_dict, None], # pressure
                                    internal.point_data['nut'][cell_dict, None] # kinematic turbulent viscosity
                                    ], axis = -1) # Internal Attributes
            
            surf_attr = np.concatenate([surf_u, # inlet_velocity
                                        surf_geom, # (=0) signed distance to aerofoil is at aerofoil 
                                        surf_normal, # normals
                                        aerofoil.point_data['U'][line_dict, :2], # velocity
                                        aerofoil.point_data['p'][line_dict, None], # pressure
                                        aerofoil.point_data['nut'][line_dict, None] # kinematic turbulent viscosity
                                        ], axis = -1) # Aerofoil Attributes

            sampled_points = cell_sampling_2d(cell_points, attr)
            surf_sampled_points = cell_sampling_1d(line_points, surf_attr)

            # Define the inputs and the targets
            pos = sampled_points[:, :2]
            init = sampled_points[:, :7]
            target = sampled_points[:, 7:]
            surf_pos = surf_sampled_points[:, :2]
            surf_init = surf_sampled_points[:, :7]
            surf_target = surf_sampled_points[:, 7:]

            # Put everything in tensor
            surf = torch.cat([torch.zeros(len(pos)), torch.ones(len(surf_pos))], dim = 0)
            pos = torch.cat([torch.tensor(pos, dtype = torch.float), torch.tensor(surf_pos, dtype = torch.float)], dim = 0) 
            x = torch.cat([torch.tensor(init, dtype = torch.float), torch.tensor(surf_init, dtype = torch.float)], dim = 0)
            y = torch.cat([torch.tensor(target, dtype = torch.float), torch.tensor(surf_target, dtype = torch.float)], dim = 0)            

        else: # Keep the mesh nodes (no sampling strategies)
            surf_bool = (internal.point_data['U'][:, 0] == 0)
            geom = -internal.point_data['implicit_distance'][:, None] # Signed distance function
            Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180
            u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(internal.point_data['U'][:, :1])
            normal = np.zeros_like(u)
            normal[surf_bool] = reorganize(aerofoil.points[:, :2], 
                                           internal.points[surf_bool, :2], 
                                           -aerofoil.point_data['Normals'][:, :2]
                                           )

            attr = np.concatenate([ u, 
                                    geom, 
                                    normal,
                                    internal.point_data['U'][:, :2], 
                                    internal.point_data['p'][:, None], 
                                    internal.point_data['nut'][:, None]
                                    ], axis = -1)

            pos = internal.points[:, :2]
            init = np.concatenate([pos, attr[:, :5]], axis = 1) # pos, inlet_vel, geom, normal
            target = attr[:, 5:] # u, p, nut

            # Put everything in tensor
            surf = torch.tensor(surf_bool)
            pos = torch.tensor(pos, dtype = torch.float)
            x = torch.tensor(init, dtype = torch.float)
            y = torch.tensor(target, dtype = torch.float)

        if norm and coef_norm is None:
            mean_in = mean_in.astype(np.single) if mean_in is not None else None
            mean_out = mean_out.astype(np.single) if mean_out is not None else None
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis = 0, dtype = np.double)
                mean_out = target.mean(axis = 0, dtype = np.double)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_in)/new_length
                mean_out += (target.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_out)/new_length
                old_length = new_length 

        data = Data(pos = pos, x = x, y = y, surf = surf.bool())
        dataset.append(data)

    if norm and coef_norm is None:
        # Compute normalization
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
                std_out = ((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_in)/new_length
                std_out += (((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_out)/new_length
                old_length = new_length
        
        std_in = np.sqrt(std_in).astype(np.single) if std_in is not None else None
        std_out = np.sqrt(std_out).astype(np.single) if std_out is not None else None

        # Normalize data wrt to its values.
        for data in dataset:
            data.x = (data.x - mean_in)/(std_in + 1e-8)
            data.y = (data.y - mean_out)/(std_out + 1e-8)

        coef_norm = (mean_in, std_in, mean_out, std_out)     
        dataset = (dataset, coef_norm)   
    
    elif coef_norm is not None:
        # Normalize data wrt to coef_norm.
        for data in dataset:
            data.x = (data.x - coef_norm[0])/(coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2])/(coef_norm[3] + 1e-8)
    
    return dataset
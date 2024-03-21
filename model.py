import torch, copy

import pandas as pd

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

# Properties of air at sea level and 293.15K
RHO = 1.184
NU = 1.56e-5
C = 346.1
P_ref = 1.013e5

class InputFeatures(torch.nn.Module):
    """Base class for input features."""

    def __init__(self) -> None:
        super().__init__()
        self.outdim = None

class FourierFeatures(InputFeatures):
    '''
    Gaussian Fourier features, as proposed in Tancik et al., NeurIPS 2020.
    '''

    def __init__(self, scale, mapdim=256, indim=3) -> None:
        super().__init__()
        self.scale = scale
        self.mapdim = mapdim
        self.outdim = 2 * mapdim
        self.indim = indim

        B = torch.randn(self.mapdim, self.indim) * self.scale**2
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = (2. * torch.pi * x) @ self.B.T
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
    
    def __repr__(self):
        return f"FourierFeatures(scale={self.scale}, mapdim={self.mapdim}, outdim={self.outdim})"


class PINN_Architecture(torch.nn.Module):
    def __init__(self, architecture):
        super(PINN_Architecture, self).__init__()
        self.layers = self._build_layers(architecture)

    def _build_layers(self, architecture):
        layers = []
        layers.append(FourierFeatures(scale=1.0, mapdim=architecture[0], indim=2))
        for i in range(1, len(architecture) - 1):
            layers.append(torch.nn.Linear(architecture[i - 1] if i == 1 else architecture[i], architecture[i]))
            layers.append(torch.nn.Softplus(beta=100))
        layers.append(torch.nn.Linear(architecture[-2], architecture[-1]))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class PINN_Phy_Bc(torch.nn.Module):
    def __init__(self, layers, df_train, df_bc, mean_variance_dict, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3, fourier_scale, fourier_mapdim):
        super(PINN_Phy_Bc, self).__init__()
        self.x = torch.tensor(df_train['x'].astype(float).values, requires_grad=True).float().unsqueeze(1)
        self.y = torch.tensor(df_train['y'].astype(float).values, requires_grad=True).float().unsqueeze(1)
        self.x_0 = torch.tensor(df_bc['x'].astype(float).values).float().unsqueeze(1)
        self.y_0 = torch.tensor(df_bc['y'].astype(float).values).float().unsqueeze(1)

        self.sdf = torch.tensor(df_train['sdf'].astype(float).values).float().unsqueeze(1)
        self.sdf_0 = torch.tensor(df_bc['sdf'].astype(float).values).float().unsqueeze(1)
                
        self.u = torch.tensor(df_train['u'].astype(float).values).float().unsqueeze(1)
        self.v = torch.tensor(df_train['v'].astype(float).values).float().unsqueeze(1)
        self.p = torch.tensor(df_train['p'].astype(float).values).float().unsqueeze(1)
        self.nut = torch.tensor(df_train['nut'].astype(float).values).float().unsqueeze(1)
        self.u_0 = torch.tensor(df_bc['u'].astype(float).values).float().unsqueeze(1)
        self.v_0 = torch.tensor(df_bc['v'].astype(float).values).float().unsqueeze(1)
        self.p_0 = torch.tensor(df_bc['p'].astype(float).values).float().unsqueeze(1)
        self.nut_0 = torch.tensor(df_bc['nut'].astype(float).values).float().unsqueeze(1)

        self.u_inlet = torch.full((len(self.x), 1), fill_value=u_inlet).float() 
        self.v_inlet = torch.full((len(self.x), 1), fill_value=v_inlet).float() 
        self.u_inlet_0 = torch.full((len(self.x_0), 1), fill_value=u_inlet).float() 
        self.v_inlet_0 = torch.full((len(self.x_0), 1), fill_value=v_inlet).float() 

        self.gamma_1 = torch.full((len(self.x), 1), fill_value=gamma_1).float() 
        self.gamma_2 = torch.full((len(self.x), 1), fill_value=gamma_2).float() 
        self.gamma_3 = torch.full((len(self.x), 1), fill_value=gamma_3).float() 
        self.gamma_1_0 = torch.full((len(self.x_0), 1), fill_value=gamma_1).float() 
        self.gamma_2_0 = torch.full((len(self.x_0), 1), fill_value=gamma_2).float() 
        self.gamma_3_0 = torch.full((len(self.x_0), 1), fill_value=gamma_3).float() 
        
        self.uv_model = PINN_Architecture([8, 128, 128, 2])
        self.p_model = PINN_Architecture([8, 64, 64, 1])
        self.nut_model = PINN_Architecture([8, 64, 64, 1])

        self.fourier_features = FourierFeatures(scale=fourier_scale, mapdim=fourier_mapdim, indim=layers[0]) # indim=2 for only 'x' and 'y'
        self.layers = [self.fourier_features.outdim] + layers[1:]
    
        self.mean_variance_dict = mean_variance_dict
        self.layers = layers
        self.loss_func = torch.nn.MSELoss()
        
        # self.adam_optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=1e-3)
        self.lbfgs_optimizer_uv = torch.optim.LBFGS([{'params': self.uv_model.parameters()}], line_search_fn='strong_wolfe')        
        self.lbfgs_optimizer_p = torch.optim.LBFGS([{'params': self.p_model.parameters()}], line_search_fn='strong_wolfe')        
        self.lbfgs_optimizer_nut = torch.optim.LBFGS([{'params': self.nut_model.parameters()}], line_search_fn='strong_wolfe')        

        self.writer = SummaryWriter(log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    def fu_fv_ic_normalized_compute(self, mean_variance_dict, u, u_x, u_y, u_xx, u_yy, v, v_x, v_y, v_xx, v_yy, p_x, p_y, nut, c_x, c_y):
        f_u = (2 * (u * mean_variance_dict['u']['var'] + mean_variance_dict['u']['mean']) * mean_variance_dict['u']['var'] * u_x) / mean_variance_dict['x']['var'] \
            + ((u * mean_variance_dict['u']['var'] + mean_variance_dict['u']['mean']) * mean_variance_dict['v']['var'] * v_y) / mean_variance_dict['y']['var'] \
            + (mean_variance_dict['u']['var'] * u_y * (v * mean_variance_dict['v']['var'] + mean_variance_dict['v']['mean'])) / mean_variance_dict['y']['var'] \
            + (mean_variance_dict['p']['var'] * p_x) / mean_variance_dict['x']['var'] \
            - (mean_variance_dict['nut']['var'] * c_x * mean_variance_dict['u']['var'] * u_x) / (mean_variance_dict['x']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['u']['var'] ** 2) * u_xx) / (mean_variance_dict['x']['var'] ** 2) \
            - (mean_variance_dict['nut']['var'] * c_y * mean_variance_dict['u']['var'] * u_y) / (mean_variance_dict['y']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['u']['var'] ** 2) * u_yy) / (mean_variance_dict['y']['var'] ** 2) 
        
        f_v = (2 * (v * mean_variance_dict['v']['var'] + mean_variance_dict['v']['mean']) * mean_variance_dict['v']['var'] * v_x) / mean_variance_dict['y']['var'] \
            + ((u * mean_variance_dict['u']['var'] + mean_variance_dict['u']['mean']) * mean_variance_dict['v']['var'] * v_x) / mean_variance_dict['x']['var'] \
            + (mean_variance_dict['u']['var'] * u_x * (v * mean_variance_dict['v']['var'] + mean_variance_dict['v']['mean'])) / mean_variance_dict['x']['var'] \
            + (mean_variance_dict['p']['var'] * p_y) / mean_variance_dict['y']['var'] \
            - (mean_variance_dict['nut']['var'] * c_x * mean_variance_dict['v']['var'] * v_x) / (mean_variance_dict['x']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['v']['var'] ** 2) * v_xx) / (mean_variance_dict['x']['var'] ** 2) \
            - (mean_variance_dict['nut']['var'] * c_y * mean_variance_dict['v']['var'] * v_y) / (mean_variance_dict['y']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['v']['var'] ** 2) * v_yy) / (mean_variance_dict['y']['var'] ** 2) 
    
        ic = ((mean_variance_dict['u']['var'] / mean_variance_dict['x']['var']) * u_x) \
            + ((mean_variance_dict['y']['var'] / mean_variance_dict['y']['var']) * v_y) # Incompressibility condition

        return f_u, f_v, ic
        
    def net_NS(self, mean_variance_dict, x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3, x_0, y_0, u_inlet_0, v_inlet_0, sdf_0, gamma_1_0, gamma_2_0, gamma_3_0):
        uv = self.uv_model(torch.cat([x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3], dim=1))
        u, v = uv.split(1, dim=1)
        uv_0 = self.uv_model(torch.cat([x_0, y_0, u_inlet_0, v_inlet_0, sdf_0, gamma_1_0, gamma_2_0, gamma_3_0], dim=1))
        u_0, v_0 = uv_0.split(1, dim=1)

        p = self.p_model(torch.cat([x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3], dim=1))
        p_0 = self.p_model(torch.cat([x_0, y_0, u_inlet_0, v_inlet_0, sdf_0, gamma_1_0, gamma_2_0, gamma_3_0], dim=1))

        nut = self.nut_model(torch.cat([x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3], dim=1))
        nut_0 = self.nut_model(torch.cat([x_0, y_0, u_inlet_0, v_inlet_0, sdf_0, gamma_1_0, gamma_2_0, gamma_3_0], dim=1))
        # c = NU + nut

        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        c_x = grad(nut, x, grad_outputs=torch.ones_like(nut), create_graph=True)[0]
        c_y = grad(nut, y, grad_outputs=torch.ones_like(nut), create_graph=True)[0]

        '''
        Laminar flow with turbulent kinematic viscosity (28) of https://doi.org/10.48550/arXiv.2212.07564.

        _uv_x = (u * v_x) + (u_x * v)
        _uv_y = (u * v_y) + (u_y * v)

        # tau_xx = grad(c*u_x, x, grad_outputs=torch.ones_like(c*u_x), create_graph=True)[0]
        tau_xx = c_x * u_x + c * u_xx
        # tau_yx = grad(c*u_y, y, grad_outputs=torch.ones_like(c*u_y), create_graph=True)[0]
        tau_yx = c_y * u_y + c * u_yy
        # tau_xy = grad(c*v_x, x, grad_outputs=torch.ones_like(c*v_x), create_graph=True)[0]
        tau_xy = c_x * v_x + c * v_xx
        # tau_yy = grad(c*v_y, y, grad_outputs=torch.ones_like(c*v_y), create_graph=True)[0]
        tau_yy = c_y * v_y + c * v_yy


        def fu_fv_ic_compute(u, u_x, v, v_y, tau_xx, tau_yx, tau_xy, tau_yy, p_x, p_y):
            f_u = (2 * u * u_x) + (u * v_y) + (u_y * v) + p_x - tau_xx - tau_yx
            f_v = (u * v_x) + (u_x * v) + (2 * v * v_y) + p_y - tau_xy - tau_yy
            ic = u_x + v_y # Incompressibility condition

            return f_u, f_v, ic
        '''
        
        f_u, f_v, ic = self.fu_fv_ic_normalized_compute(mean_variance_dict, u, u_x, u_y, u_xx, u_yy, v, v_x, v_y, v_xx, v_yy, p_x, p_y, nut, c_x, c_y)

        return u_0, v_0, p_0, nut_0, f_u, f_v, ic

    def forward(self, mean_variance_dict, x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3, x_0, y_0, u_inlet_0, v_inlet_0, sdf_0, gamma_1_0, gamma_2_0, gamma_3_0):
        u_0_pred, v_0_pred, p_0_pred, nut_0_pred, \
        f_u_pred, f_v_pred, ic_pred = self.net_NS(
                                                    mean_variance_dict,
                                                    x, y, u_inlet, v_inlet, 
                                                    sdf, gamma_1, gamma_2, gamma_3, 
                                                    x_0, y_0, u_inlet_0, v_inlet_0, 
                                                    sdf_0, gamma_1_0, gamma_2_0, gamma_3_0,
                                                )
        
        f_u_loss, f_v_loss = self.loss_func(f_u_pred, torch.zeros_like(f_u_pred)), self.loss_func(f_v_pred, torch.zeros_like(f_v_pred))
        rans_loss = f_u_loss + f_v_loss # Reynold-Average NS total loss

        self.mean_variance_dict = mean_variance_dict
        
        u_0_loss = self.loss_func(self.u_0, u_0_pred)
        v_0_loss = self.loss_func(self.v_0, v_0_pred)

        uv_0_loss = u_0_loss + v_0_loss
        p_0_loss = self.loss_func(self.p_0, p_0_pred)
        nut_0_loss = self.loss_func(self.nut_0, nut_0_pred)

        ic_loss = self.loss_func(ic_pred, torch.zeros_like(ic_pred))

        uv_loss = uv_0_loss + ic_loss + rans_loss
        p_loss = p_0_loss + rans_loss
        nut_loss = nut_0_loss + rans_loss

        return uv_loss, p_loss, nut_loss, u_0_loss, v_0_loss, p_0_loss, nut_0_loss, ic_loss, f_u_loss, f_v_loss

    def train(self, nIter):
        # Temporary storage for loss values for logging purposes
        self.temp_losses = None

        def closure_uv():
            self.lbfgs_optimizer_uv.zero_grad()
            # Compute the forward pass and losses
            uv_loss, p_loss, nut_loss, \
            u_0_loss, v_0_loss, p_0_loss, nut_0_loss, \
            ic_loss, f_u_loss, f_v_loss = self.forward(
                                                    self.mean_variance_dict,
                                                    self.x, self.y, self.u_inlet, self.v_inlet, 
                                                    self.sdf, self.gamma_1, self.gamma_2, self.gamma_3,
                                                    self.x_0, self.y_0, self.u_inlet_0, self.v_inlet_0, 
                                                    self.sdf_0, self.gamma_1_0, self.gamma_2_0, self.gamma_3_0,
                                                )

            uv_loss.backward()

            # Store losses in the model for access outside the closure
            self.temp_losses =  (
                                    ('uv_loss', uv_loss), ('p_loss', p_loss), ('nut_loss', nut_loss), 
                                    ('u_0_loss', u_0_loss), ('v_0_loss', v_0_loss), ('p_0_loss', p_0_loss), 
                                    ('nut_0_loss', nut_0_loss), ('ic_loss', ic_loss), ('f_u_loss', f_u_loss), 
                                    ('f_v_loss', f_v_loss)
                                )
            return uv_loss
        
        def closure_p():
            self.lbfgs_optimizer_p.zero_grad()
            # Compute the forward pass and losses
            uv_loss, p_loss, nut_loss, \
            u_0_loss, v_0_loss, p_0_loss, nut_0_loss, \
            ic_loss, f_u_loss, f_v_loss = self.forward(
                                                    self.mean_variance_dict,
                                                    self.x, self.y, self.u_inlet, self.v_inlet, 
                                                    self.sdf, self.gamma_1, self.gamma_2, self.gamma_3,
                                                    self.x_0, self.y_0, self.u_inlet_0, self.v_inlet_0, 
                                                    self.sdf_0, self.gamma_1_0, self.gamma_2_0, self.gamma_3_0,
                                                )

            p_loss.backward()
            return p_loss
        
        def closure_nut():
            self.lbfgs_optimizer_nut.zero_grad()
            # Compute the forward pass and losses
            uv_loss, p_loss, nut_loss, \
            u_0_loss, v_0_loss, p_0_loss, nut_0_loss, \
            ic_loss, f_u_loss, f_v_loss = self.forward(
                                                    self.mean_variance_dict,
                                                    self.x, self.y, self.u_inlet, self.v_inlet, 
                                                    self.sdf, self.gamma_1, self.gamma_2, self.gamma_3,
                                                    self.x_0, self.y_0, self.u_inlet_0, self.v_inlet_0, 
                                                    self.sdf_0, self.gamma_1_0, self.gamma_2_0, self.gamma_3_0,
                                                )

            nut_loss.backward()

            return nut_loss

        for it in range(nIter):
            optimizer_uv = self.lbfgs_optimizer_uv
            optimizer_p = self.lbfgs_optimizer_p
            optimizer_nut = self.lbfgs_optimizer_nut
            optimizer_uv.step(closure_uv)
            optimizer_p.step(closure_p)
            optimizer_nut.step(closure_nut)

            if it % 10 == 0: # show iterations
                for name, value in self.temp_losses:
                    print(f"{name}: {value.item()}")


    def predict(self, df_test, df_bc, mean_variance_dict, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3, fourier_scale, fourier_mapdim):
        x = torch.tensor(df_test['x'].astype(float).values, requires_grad=True).float()
        y = torch.tensor(df_test['y'].astype(float).values, requires_grad=True).float()
        x_0 = torch.tensor(df_bc['x'].astype(float).values).float()
        y_0 = torch.tensor(df_bc['y'].astype(float).values).float()

        sdf = torch.tensor(df_test['sdf'].astype(float).values).float()
        sdf_0 = torch.tensor(df_bc['sdf'].astype(float).values).float()
                
        u_inlet = torch.full_like(x, fill_value=u_inlet).float()
        v_inlet = torch.full_like(x, fill_value=v_inlet).float()
        u_inlet_0 = torch.full_like(x_0, fill_value=u_inlet).float()
        v_inlet_0 = torch.full_like(x_0, fill_value=v_inlet).float()

        gamma_1 = torch.full_like(x, fill_value=gamma_1).float()
        gamma_2 = torch.full_like(x, fill_value=gamma_2).float()
        gamma_3 = torch.full_like(x, fill_value=gamma_3).float()
        gamma_1_0 = torch.full_like(x_0, fill_value=gamma_1).float()
        gamma_2_0 = torch.full_like(x_0, fill_value=gamma_2).float()
        gamma_3_0 = torch.full_like(x_0, fill_value=gamma_3).float()

        uv_star = self.uv_model(df_test, df_bc, mean_variance_dict, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3, fourier_scale, fourier_mapdim)
        p_star = self.p_model(df_test, df_bc, mean_variance_dict, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3, fourier_scale, fourier_mapdim)
        nut_star = self.nut_model(df_test, df_bc, mean_variance_dict, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3, fourier_scale, fourier_mapdim)

        u_star, v_star = uv_star.split(dim=1)

        return u_star.detach().numpy(), v_star.detach().numpy(), p_star.detach().numpy(), nut_star.detach().numpy()

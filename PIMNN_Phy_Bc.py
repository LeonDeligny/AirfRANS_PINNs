import torch, os

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

# from log import nan_gradients

# Properties of air at sea level and 293.15K
RHO = 1.184
NU = 1.56e-5
C = 346.1
P_ref = 1.013e5


# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class InputFeatures(torch.nn.Module):
    """Base class for input features."""

    def __init__(self) -> None:
        super().__init__()
        self.outdim = None

class FourierFeatures(InputFeatures):
    '''
    Gaussian Fourier features, as proposed in Tancik et al., NeurIPS 2020.
    '''

    def __init__(self, scale, mapdim, indim) -> None:
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
    def __init__(self, architecture, fourier_scale, fourier_mapdim, indim):
        super(PINN_Architecture, self).__init__()
        self.fourier_scale = fourier_scale
        self.fourier_mapdim = fourier_mapdim
        self.indim = indim
        self.layers = self._build_layers(architecture)

    def _build_layers(self, architecture):
        fourier_features = FourierFeatures(self.fourier_scale, self.fourier_mapdim, self.indim)
        model_layers = [fourier_features]
        model_layers.append(torch.nn.Linear(fourier_features.outdim, architecture[1]))
        # Add remaining layers
        for i in range(1, len(architecture)-1):
            model_layers.append(torch.nn.Softplus(beta=100))
            model_layers.append(torch.nn.Linear(architecture[i], architecture[i+1]))
        
        return torch.nn.Sequential(*model_layers)
    
    def forward(self, x):
        return self.layers(x)


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.log1p(torch.abs(y_pred - y_true)))


class PIMNN_Phy_Bc(torch.nn.Module):
    def __init__(self, df_train, mean_variance_dict, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3):
        super(PIMNN_Phy_Bc, self).__init__()
        self.x = torch.tensor(df_train['x'].astype(float).values).float().unsqueeze(1).to(device)
        self.y = torch.tensor(df_train['y'].astype(float).values).float().unsqueeze(1).to(device)
        self.sdf = torch.tensor(df_train['sdf'].astype(float).values).float().unsqueeze(1).to(device)

        self.u = torch.tensor(df_train['u'].astype(float).values).float().unsqueeze(1).to(device)
        self.v = torch.tensor(df_train['v'].astype(float).values).float().unsqueeze(1).to(device)
        self.nut = torch.tensor(df_train['nut'].astype(float).values).float().unsqueeze(1).to(device)
        self.p = torch.tensor(df_train['p'].astype(float).values).float().unsqueeze(1).to(device)

        self.u_inlet = torch.full((len(self.x), 1), fill_value=u_inlet).float().to(device)
        self.v_inlet = torch.full((len(self.x), 1), fill_value=v_inlet).float().to(device)
        self.gamma_1 = torch.full((len(self.x), 1), fill_value=gamma_1).float().to(device)
        self.gamma_2 = torch.full((len(self.x), 1), fill_value=gamma_2).float().to(device)
        self.gamma_3 = torch.full((len(self.x), 1), fill_value=gamma_3).float().to(device)
        
        u_layers = [8, 512, 512, 512, 512, 512, 512, 1]
        v_layers = [8, 512, 512, 512, 512, 512, 512, 1]
        p_layers = [8, 512, 512, 512, 512, 512, 512, 1]
        nut_layers = [8, 512, 512, 512, 512, 512, 512, 1]

        print(f"u layers: {u_layers}")
        print(f"v layers: {v_layers}")
        print(f"p layers: {p_layers}")
        print(f"nut layers: {nut_layers}")

        self.u_model = self.create_model(u_layers)
        self.v_model = self.create_model(v_layers)
        self.p_model = self.create_model(p_layers)
        self.nut_model = self.create_model(nut_layers)
    
        self.mean_variance_dict = mean_variance_dict
        self.mse_loss = torch.nn.MSELoss()
        self.log_loss = torch.nn.MSELoss()

        self.lbfgs_optimizer_u = torch.optim.LBFGS([{'params': self.u_model.parameters()}], line_search_fn='strong_wolfe') 
        self.lbfgs_optimizer_v = torch.optim.LBFGS([{'params': self.v_model.parameters()}], line_search_fn='strong_wolfe')
        self.lbfgs_optimizer_p = torch.optim.LBFGS([{'params': self.p_model.parameters()}], line_search_fn='strong_wolfe')       
        self.lbfgs_optimizer_nut = torch.optim.LBFGS([{'params': self.nut_model.parameters()}], line_search_fn='strong_wolfe')   
        
        self.writer = SummaryWriter(log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    def create_model(self, type_layers):
        layers = []
        for i in range(len(type_layers) - 1):
            layers.append(torch.nn.Linear(type_layers[i], type_layers[i+1]))
            if i != len(type_layers) - 2:
                layers.append(torch.nn.Softplus(100))
        return torch.nn.Sequential(*layers)
    
    def fu_fv_ic_normalized_compute(self, mean_variance_dict, u, u_x, u_y, u_xx, u_yy, v, v_x, v_y, v_xx, v_yy, p_x, p_y, nut, nut_x, nut_y):
        f_u = (2 * (u * mean_variance_dict['u']['var'] + mean_variance_dict['u']['mean']) * mean_variance_dict['u']['var'] * u_x) / mean_variance_dict['x']['var'] \
            + ((u * mean_variance_dict['u']['var'] + mean_variance_dict['u']['mean']) * mean_variance_dict['v']['var'] * v_y) / mean_variance_dict['y']['var'] \
            + (mean_variance_dict['u']['var'] * u_y * (v * mean_variance_dict['v']['var'] + mean_variance_dict['v']['mean'])) / mean_variance_dict['y']['var'] \
            + (mean_variance_dict['p']['var'] * p_x) / mean_variance_dict['x']['var'] \
            - (mean_variance_dict['nut']['var'] * nut_x * mean_variance_dict['u']['var'] * u_x) / (mean_variance_dict['x']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['u']['var'] ** 2) * u_xx) / (mean_variance_dict['x']['var'] ** 2) \
            - (mean_variance_dict['nut']['var'] * nut_y * mean_variance_dict['u']['var'] * u_y) / (mean_variance_dict['y']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['u']['var'] ** 2) * u_yy) / (mean_variance_dict['y']['var'] ** 2) 
        
        f_v = (2 * (v * mean_variance_dict['v']['var'] + mean_variance_dict['v']['mean']) * mean_variance_dict['v']['var'] * v_x) / mean_variance_dict['y']['var'] \
            + ((u * mean_variance_dict['u']['var'] + mean_variance_dict['u']['mean']) * mean_variance_dict['v']['var'] * v_x) / mean_variance_dict['x']['var'] \
            + (mean_variance_dict['u']['var'] * u_x * (v * mean_variance_dict['v']['var'] + mean_variance_dict['v']['mean'])) / mean_variance_dict['x']['var'] \
            + (mean_variance_dict['p']['var'] * p_y) / mean_variance_dict['y']['var'] \
            - (mean_variance_dict['nut']['var'] * nut_x * mean_variance_dict['v']['var'] * v_x) / (mean_variance_dict['x']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['v']['var'] ** 2) * v_xx) / (mean_variance_dict['x']['var'] ** 2) \
            - (mean_variance_dict['nut']['var'] * nut_y * mean_variance_dict['v']['var'] * v_y) / (mean_variance_dict['y']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['v']['var'] ** 2) * v_yy) / (mean_variance_dict['y']['var'] ** 2) 
    
        ic = ((mean_variance_dict['u']['var'] / mean_variance_dict['x']['var']) * u_x) \
            + ((mean_variance_dict['v']['var'] / mean_variance_dict['y']['var']) * v_y) # Incompressibility condition

        return f_u, f_v, ic
        
    def net_NS(self, mean_variance_dict, x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3):
        inputs = torch.cat([x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3], dim=1)

        u = self.u_model(inputs)
        v = self.v_model(inputs)
        p = self.p_model(inputs)
        nut = self.nut_model(inputs)

        '''
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

        nut_x = grad(nut, x, grad_outputs=torch.ones_like(nut), create_graph=True)[0]
        nut_y = grad(nut, y, grad_outputs=torch.ones_like(nut), create_graph=True)[0]
        
        f_u, f_v, ic = self.fu_fv_ic_normalized_compute(mean_variance_dict, u, u_x, u_y, u_xx, u_yy, v, v_x, v_y, v_xx, v_yy, p_x, p_y, nut, nut_x, nut_y)
        bc = self.bc_normalized_compute(mean_variance_dict, p_x_a, p_y_a, x_n_a, y_n_a)
        '''

        return u, v, p, nut

    def forward(self, mean_variance_dict, x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3):
        u_pred, v_pred, p_pred, nut_pred = self.net_NS( 
                                                    mean_variance_dict, 
                                                    x, y, u_inlet, v_inlet, 
                                                    sdf, gamma_1, gamma_2, gamma_3
                                                        )
        
    
        '''
        f_u_loss, f_v_loss = self.mse_loss(f_u_pred, torch.zeros_like(f_u_pred)), self.mse_loss(f_v_pred, torch.zeros_like(f_v_pred))
        # rans_loss = f_u_loss + f_v_loss
        f_u_loss_norm, f_v_loss_norm = f_u_loss / (torch.abs(f_u_pred).mean() + 1e-8), f_v_loss / (torch.abs(f_v_pred).mean() + 1e-8)
        rans_loss_norm = f_u_loss_norm + f_v_loss_norm

        ic_loss = self.mse_loss(ic_pred, torch.zeros_like(ic_pred))
        ic_loss_norm = ic_loss / (torch.abs(ic_pred).mean() + 1e-8)

        bc_loss = self.mse_loss(bc_pred, torch.zeros_like(bc_pred))
        bc_loss_norm = bc_loss / (torch.abs(bc_loss).mean() + 1e-8)
        '''
        
        u_train_loss = self.mse_loss(self.u, u_pred) 
        v_train_loss = self.mse_loss(self.v, v_pred)
        p_train_loss = self.mse_loss(self.p, p_pred) 
        nut_train_loss = self.log_loss(self.nut, nut_pred)

        '''
        u_loss = u_f_loss + u_a_loss + ic_loss_norm + rans_loss_norm # ic_loss + rans_loss # self.mse_loss(self.u, u_pred) 
        v_loss = v_f_loss + v_a_loss + ic_loss_norm + rans_loss_norm # ic_loss + rans_loss # self.mse_loss(self.v, v_pred)
        p_loss = p_f_loss + bc_loss_norm + rans_loss_norm # rans_loss # self.mse_loss(self.p, p_pred) 
        nut_loss = nut_f_loss + nut_a_loss + rans_loss_norm # rans_loss # self.mse_loss(self.nut, nut_pred)
        '''

        return u_train_loss, v_train_loss, p_train_loss, nut_train_loss


    def train(self, nIter, checkpoint_path='path_to_checkpoint.pth'):
        # Temporary storage for loss values for logging purposes
        self.temp_losses = {}
        self.display = {}

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            
            self.u_model.load_state_dict(checkpoint['u_model_state_dict'])
            self.lbfgs_optimizer_u.load_state_dict(checkpoint['lbfgs_optimizer_u_state_dict'])
            
            self.v_model.load_state_dict(checkpoint['v_model_state_dict'])
            self.lbfgs_optimizer_v.load_state_dict(checkpoint['lbfgs_optimizer_v_state_dict'])
            
            self.p_model.load_state_dict(checkpoint['p_model_state_dict'])
            self.lbfgs_optimizer_p.load_state_dict(checkpoint['lbfgs_optimizer_p_state_dict'])
            
            self.nut_model.load_state_dict(checkpoint['nut_model_state_dict'])
            self.lbfgs_optimizer_nut.load_state_dict(checkpoint['lbfgs_optimizer_nut_state_dict'])
            
            # Restore the RNG state
            torch.set_rng_state(checkpoint['rng_state'])

            # If you're resuming training and want to start from the next iteration,
            # make sure to load the last iteration count and add one
            start_iteration = checkpoint.get('iterations', 0) + 1
            print(f"Resuming from iteration {start_iteration}")
        else:
            print(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
            start_iteration = 0

        def compute_losses():
            # Compute all losses
            losses = self.forward(self.mean_variance_dict,
                                self.x, self.y, self.u_inlet, self.v_inlet, 
                                self.sdf, self.gamma_1, self.gamma_2, self.gamma_3
                                )

            # Unpack the losses and store them in a dictionary for easy access
            (u_loss, v_loss, p_loss, nut_loss) = losses

            self.temp_losses = {'u_loss': u_loss, 'v_loss': v_loss, 'p_loss': p_loss, 'nut_loss': nut_loss}

            self.display = {
                            'u_loss': u_loss, 'v_loss': v_loss, 'p_loss': p_loss, 'nut_loss': nut_loss,
                            #'u_train_loss': u_train_loss, 'v_train_loss': v_train_loss, 'p_train_loss': p_train_loss, 'nut_train_loss': nut_train_loss,
                            # 'u_f_loss': u_f_loss, 'v_f_loss': v_f_loss, 'p_f_loss': p_f_loss, 'nut_f_loss': nut_f_loss, 
                            # 'u_a_loss': u_a_loss, 'v_a_loss': v_a_loss, 'nut_a_loss': nut_a_loss, 
                            #'f_u_loss': f_u_loss, 'f_v_loss': f_v_loss, 'ic_loss': ic_loss, 'bc_loss': bc_loss
                        }


        for it in range(start_iteration, nIter + start_iteration):
            # Define closure for LBFGS optimization for u
            def closure_u():
                self.lbfgs_optimizer_u.zero_grad()
                compute_losses()
                self.temp_losses['u_loss'].backward()
                # nan_gradients(self.u_model)

                return self.temp_losses['u_loss']

            # Perform LBFGS step for u
            self.lbfgs_optimizer_u.step(closure_u)

            # Define closure for LBFGS optimization for v
            def closure_v():
                self.lbfgs_optimizer_v.zero_grad()
                compute_losses()
                self.temp_losses['v_loss'].backward()
                # nan_gradients(self.v_model)

                return self.temp_losses['v_loss']

            # Perform LBFGS step for v
            self.lbfgs_optimizer_v.step(closure_v)

            # Define closure for LBFGS optimization for p
            def closure_p():
                self.lbfgs_optimizer_p.zero_grad()
                compute_losses()
                self.temp_losses['p_loss'].backward()
                # nan_gradients(self.p_model)

                return self.temp_losses['p_loss']

            # Perform LBFGS step for p
            self.lbfgs_optimizer_p.step(closure_p)

            # Define closure for LBFGS optimization for nut
            def closure_nut():
                self.lbfgs_optimizer_nut.zero_grad()
                compute_losses()
                self.temp_losses['nut_loss'].backward()
                # nan_gradients(self.nut_model)

                return self.temp_losses['nut_loss']

            # Perform LBFGS step for nut
            self.lbfgs_optimizer_nut.step(closure_nut)
            
            if it % 2 == 0:
                print(f"Iteration: {it}")
            if it % 10 == 0:  # Print losses every 10 iterations
                for name, value in self.display.items():
                    print(f"{name}: {value.item()}")

                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                

                checkpoint = {
                    'u_model_state_dict': self.u_model.state_dict(),
                    'lbfgs_optimizer_u_state_dict': self.lbfgs_optimizer_u.state_dict(),

                    'v_model_state_dict': self.v_model.state_dict(),
                    'lbfgs_optimizer_v_state_dict': self.lbfgs_optimizer_v.state_dict(),

                    'p_model_state_dict': self.p_model.state_dict(),
                    'lbfgs_optimizer_p_state_dict': self.lbfgs_optimizer_p.state_dict(),

                    'nut_model_state_dict': self.nut_model.state_dict(),
                    'lbfgs_optimizer_nut_state_dict': self.lbfgs_optimizer_nut.state_dict(),

                    'iterations': it,

                    'rng_state': torch.get_rng_state(),
                }

                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to '{checkpoint_path}' at iteration {it}")



    def predict(self, df_test, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3):
        x_star = torch.tensor(df_test['x'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        y_star = torch.tensor(df_test['y'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)

        sdf_star = torch.tensor(df_test['sdf'].astype(float).values).float().unsqueeze(1).to(device)
                
        u_inlet_star = torch.full((len(x_star), 1), fill_value=u_inlet).float().to(device)
        v_inlet_star = torch.full((len(x_star), 1), fill_value=v_inlet).float().to(device)

        gamma_1_star = torch.full((len(x_star), 1), fill_value=gamma_1).float().to(device)
        gamma_2_star = torch.full((len(x_star), 1), fill_value=gamma_2).float().to(device)
        gamma_3_star = torch.full((len(x_star), 1), fill_value=gamma_3).float().to(device)

        inputs = torch.cat([x_star, y_star, u_inlet_star, v_inlet_star, sdf_star, gamma_1_star, gamma_2_star, gamma_3_star], dim=1)

        u_star = self.u_model(inputs)
        v_star = self.v_model(inputs)
        p_star = self.p_model(inputs)
        nut_star = self.nut_model(inputs)

        return u_star.cpu().detach().numpy(), v_star.cpu().detach().numpy(), p_star.cpu().detach().numpy(), nut_star.cpu().detach().numpy()
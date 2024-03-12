import torch
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

# Properties of air at sea level and 293.15K
RHO = 1.184
NU = 1.56e-5
C = 346.1
P_ref = 1.013e5

# iteration to switch from Adam to L-BFGS
it_Adam_to_LBGFS = 0

class PhysicsInformedNN(torch.nn.Module):
    def __init__(self, layers, coef_norm, u, v, p, nut, x, y, x_normal, y_normal, sdf, gamma_1, gamma_2, gamma_3):
        super(PhysicsInformedNN, self).__init__()
        # position of the particle
        self.x = torch.tensor(x, requires_grad=True).float()
        self.y = torch.tensor(y, requires_grad=True).float()
    
        # normal to sdf
        self.x_normal = torch.tensor(x_normal).float()
        self.y_normal = torch.tensor(y_normal).float()  

        # signed distance function
        self.sdf = torch.tensor(sdf).float()
        
        # initial vorticty
        # self.u_inlet = torch.tensor(u_inlet).float()
        # self.v_inlet = torch.tensor(v_inlet).float()

        # airfoil structure
        self.gamma_1 = torch.tensor(gamma_1).float()
        self.gamma_2 = torch.tensor(gamma_2).float()
        self.gamma_3 = torch.tensor(gamma_3).float()

        # self.u = torch.nn.Parameter(torch.zeros(1))
        # self.v = torch.nn.Parameter(torch.zeros(1))
        # self.p = torch.nn.Parameter(torch.zeros(1))
        
        self.layers = layers

        self.coef_norm = coef_norm

        self.u_0 = torch.tensor(u).float()
        self.v_0 = torch.tensor(v).float()
        self.nut_0 = torch.tensor(nut).float()
        self.p_0 = torch.tensor(p).float()

        self.loss_func = torch.nn.MSELoss()
        
        self.model = self.create_model()
        
        # self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=0.01)        
        # self.optimizer = torch.optim.LBFGS(self.model.parameters())
        self.adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.lbfgs_optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1e-4, max_iter=20, history_size=100)        
        
        self.writer = SummaryWriter(log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    def create_model(self):
        layers = []
        for i in range(len(self.layers) - 1):
            layers.append(torch.nn.Linear(self.layers[i], self.layers[i+1]))
            if i != len(self.layers) - 2:
                layers.append(torch.nn.Tanh())

        return torch.nn.Sequential(*layers)

    def net_NS(self, x, y, x_normal, y_normal, sdf, gamma_1, gamma_2, gamma_3): # x, y, x_normal, y_normal, sdf, nut, gamma_1, gamma_2, gamma_3
        u_v_p_nut = self.model(torch.cat([x, y, x_normal, y_normal, sdf, gamma_1, gamma_2, gamma_3], dim=1))
        
        u = u_v_p_nut[:, 0:1]
        v = u_v_p_nut[:, 1:2]
        p = u_v_p_nut[:, 2:3]

        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]


        '''
        Laminar flow without turbulent kinematic viscosity (Navier-Stockes Equations)
        '''

        '''
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_xy = grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        v_yx = grad(v_x, y, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

        f_u = (u * u_x + v * u_y) + p_x - ( NU * (u_xx + u_yy + u_xx + v_yx) )
        f_v = (u * v_x + v * v_y) + p_y - ( NU * (v_xx + v_yy + u_xy + v_xx) )
        '''

        '''
        Laminar flow with turbulent kinematic viscosity (28) of https://doi.org/10.48550/arXiv.2212.07564.

        \part_x(uv) + \part_y(uv) + \part_x(p) + \part_x( (NU + nut)\part_x(u) ) + \part_y( (NU + nut) \part_y(u) ) = 0
        \part_x(uv) + \part_y(uv) + \part_y(p) + \part_x( (NU + nut)\part_x(v) ) + \part_y( (NU + nut) \part_y(v) ) = 0
        '''

        nut = u_v_p_nut[:, 3:4]
        uv = u*v
        c = NU + nut

        _uv_x = grad(uv, x, grad_outputs=torch.ones_like(uv), create_graph=True)[0]
        _uv_y = grad(uv, y, grad_outputs=torch.ones_like(uv), create_graph=True)[0]

        _uv_x_y_ = _uv_x + _uv_y

        tau_xx = grad(c*u_x, x, grad_outputs=torch.ones_like(c*u_x), create_graph=True)[0]
        tau_yx = grad(c*u_y, y, grad_outputs=torch.ones_like(c*u_y), create_graph=True)[0]
        tau_xy = grad(c*v_x, x, grad_outputs=torch.ones_like(c*v_x), create_graph=True)[0]
        tau_yy = grad(c*v_y, y, grad_outputs=torch.ones_like(c*v_y), create_graph=True)[0]


        f_u = _uv_x_y_ + p_x - tau_xx - tau_yx
        f_v = _uv_x_y_ + p_y - tau_xy - tau_yy

        '''
        Incompressibility condition
        '''

        ic = u_x + v_y

        return u, v, p, nut, f_u, f_v, ic

    def forward(self, coef_norm, x, y, x_normal, y_normal, sdf, gamma_1, gamma_2, gamma_3):
        u_pred, v_pred, p_pred, nut_pred, f_u_pred, f_v_pred, ic_pred = self.net_NS(x, y, x_normal, y_normal, sdf, gamma_1, gamma_2, gamma_3)
        
        f_u_loss, f_v_loss = self.loss_func(f_u_pred, torch.zeros_like(f_u_pred)), self.loss_func(f_v_pred, torch.zeros_like(f_v_pred))
        
        rans_loss = f_u_loss + f_v_loss # Reynold-Average NS loss

        aerofoil_bc =   ( ((u_pred * (coef_norm[3][0] + 1e-8)) + coef_norm[2][0]) \
                        * ((x_normal * (coef_norm[1][5] + 1e-8)) + coef_norm[0][5]) ) \
                        + ( ((v_pred * (coef_norm[3][1] + 1e-8)) + coef_norm[2][1]) \
                        * ((y_normal * (coef_norm[1][6] + 1e-8)) + coef_norm[0][6]) ) # u . n = 0 on aerofoil surface

        aerofoil_bc_loss = self.loss_func(aerofoil_bc, torch.zeros_like(aerofoil_bc))
        
        ''' 
        mask = ( ((x * (coef_norm[1][0] + 1e-8)) + coef_norm[0][0]) >= -2 -0.1618361) & (((x * (coef_norm[1][0] + 1e-8)) + coef_norm[0][0]) <= -2 + 0.1618361)
        inlet_bc_loss = torch.where(mask,   self.loss_func(self.u_0, u_pred) \
                                            + self.loss_func(self.v_0, v_pred) \
                                            + self.loss_func(self.nut_0, nut_pred) 
                                            + self.loss_func(self.p_0, p_pred), torch.tensor(0.0).to(u_pred.device)
                                    ).mean() # Boundary condition with inlet, viscosity and pressure initialization.
        
        '''

        u_loss = self.loss_func(self.u_0, u_pred)
        v_loss = self.loss_func(self.v_0, v_pred)
        nut_loss = self.loss_func(self.nut_0, nut_pred)
        p_loss = self.loss_func(self.p_0, p_pred)

        inlet_bc_loss = u_loss + v_loss + p_loss + nut_loss

        ic_loss = self.loss_func(ic_pred, torch.zeros_like(ic_pred))
        
        total_loss = rans_loss + aerofoil_bc_loss + inlet_bc_loss + ic_loss

        return total_loss, rans_loss, aerofoil_bc_loss, inlet_bc_loss, ic_loss, u_loss, v_loss, p_loss, nut_loss

    def train(self, nIter):
        # Temporary storage for loss values for logging purposes
        self.temp_losses = None

        def closure():
            self.lbfgs_optimizer.zero_grad()
            # Compute the forward pass and losses
            total_loss, rans_loss, aerofoil_bc_loss, inlet_bc_loss,ic_loss, u_loss, v_loss, p_loss, nut_loss = self.forward(self.coef_norm, self.x, self.y, self.x_normal, self.y_normal, self.sdf, self.gamma_1, self.gamma_2, self.gamma_3)
            total_loss.backward()
            # Store losses in the model for access outside the closure
            self.temp_losses = (total_loss, rans_loss, aerofoil_bc_loss, inlet_bc_loss, ic_loss, u_loss, v_loss, p_loss, nut_loss)
            
            return total_loss # Only return the total_loss, as required by LBFGS

        for it in range(nIter):
            optimizer = self.adam_optimizer if it < it_Adam_to_LBGFS else self.lbfgs_optimizer

            if it >= it_Adam_to_LBGFS:
                optimizer.step(closure)
            else:
                # For Adam, perform a regular optimization step without closure
                optimizer.zero_grad()
                total_loss, rans_loss, aerofoil_bc_loss, inlet_bc_loss, ic_loss, u_loss, v_loss, p_loss, nut_loss = self.forward(self.coef_norm, self.x, self.y, self.x_normal, self.y_normal, self.sdf, self.gamma_1, self.gamma_2, self.gamma_3)
                self.temp_losses = (total_loss, rans_loss, aerofoil_bc_loss, inlet_bc_loss, ic_loss, u_loss, v_loss, p_loss, nut_loss)
                total_loss.backward()
                optimizer.step()

            # Access the stored loss values for logging; no need to recompute the forward pass
            if self.temp_losses:    
                total_loss, rans_loss, aerofoil_bc_loss, inlet_bc_loss, ic_loss, u_loss, v_loss, p_loss, nut_loss = self.temp_losses

            if it % 1 == 0: # show iterations
                print(f'It: {it}, Total Loss: {total_loss.item()}')
                print(f'rans_loss: {rans_loss.item()}')
                print(f'aerofoil_bc_loss: {aerofoil_bc_loss.item()}')
                print(f'inlet_bc_loss: {inlet_bc_loss.item()}')
                print(f'ic_loss: {ic_loss.item()}')

                print(f'u_loss: {u_loss.item()}')
                print(f'v_loss: {v_loss.item()}')
                print(f'p_loss: {p_loss.item()}')
                print(f'nut_loss: {nut_loss.item()}')

                self.writer.add_scalar('Total Loss', total_loss.item(), it)

    def predict(self, x_star, y_star, x_normal_star, y_normal_star, sdf_star, gamma_1_star, gamma_2_star, gamma_3_star):
        x_star = torch.tensor(x_star, requires_grad=True)
        y_star = torch.tensor(y_star, requires_grad=True)
        x_normal_star = torch.tensor(x_normal_star)
        y_normal_star = torch.tensor(y_normal_star)
        sdf_star = torch.tensor(sdf_star)
        gamma_1_star = torch.tensor(gamma_1_star)
        gamma_2_star = torch.tensor(gamma_2_star)
        gamma_3_star = torch.tensor(gamma_3_star)

        u_star, v_star, p_star, nut_star, _, _, _ = self.net_NS(x_star, y_star, x_normal_star, y_normal_star, sdf_star, gamma_1_star, gamma_2_star, gamma_3_star)

        return u_star.detach().numpy(), v_star.detach().numpy(), p_star.detach().numpy(), nut_star.detach().numpy()

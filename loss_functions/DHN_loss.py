import path
import sys
 
# directory reach
directory = path.Path(__file__).absolute()
 
# setting path
sys.path.append(directory.parent.parent)

import torch
from config import device
from assistive_functions import to_tensor



class DHNLoss():
    def __init__(
        self, R, u_min, u_max, x_min,x_max,peak = False, renewable_fossil_peak = False,
        alpha_xl=None, alpha_xh=None,
        loss_bound=None, sat_bound=None,
    ):
        
        self.peak = peak

        self.umin = to_tensor(u_min).to(device)
        self.umax = to_tensor(u_max).to(device)
        self.xmin = to_tensor(x_min).to(device)
        self.xmax = to_tensor(x_max).to(device)

        self.loss_bound = loss_bound
        self.sat_bound = sat_bound

        self.alpha_xl = alpha_xl
        self.alpha_xh = alpha_xh

        #Weight of the base lost
        self.R = R
        self.R = to_tensor(self.R)
        if isinstance(self.R, torch.Tensor):     # cast to device if is not a scalar
            self.R = self.R.to(device)
        assert (not hasattr(self.R, "__len__")) or len(self.R.shape) == 2  # int or square matrix
        print(self.R)


        #Create tariff over time vector
        high = [3]*12
        low = [1]*12
        self.tariff = torch.tensor(high+low).to(device) # not used



    def forward(self, xs, us):      ## removed dxref
        """
        Compute loss.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, in_dim)

        Return:
            - loss of shape (1, 1).
        """

        # batch
        x_batch = xs.reshape(*xs.shape,1)
        u_batch = us.reshape(*us.shape, 1)
        # dxref = dxref.reshape(*dxref.shape, 1)

        # loss states = 1/T sum_{t=1}^T (x_t-xbar)^T Q (x_t-xbar)
        
        # loss control actions = 1/T sum_{t=1}^T u_t^T R u_t

        if self.peak:
            u_b = u_batch.clone()
            for i in range(13):
                u_b[:,i,:,:] = u_b[:,i,:,:]*10
            loss_u = torch.sum(u_b, 1) / x_batch.shape[1]    # average over the time horizon. shape = (S, 1, 1)
        
        else: 
            uTRu = self.R * torch.matmul(
                u_batch.transpose(-1, -2),
                u_batch
            )   # shape = (S, T, 1, 1)
            loss_u = torch.sum(uTRu, 1) / x_batch.shape[1] 

        # if self.renewable_fossil_peak:
        #     heat_demand =  [30,20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 90, 80, 70, 60, 50, 60, 80, 100, 90, 80, 70, 50, 40]
        #     fossil_rescale = 0.1
        #     fossil_price = fossil_rescale * heat_demand

        #     renewable_avail = [1, 2, 3, 5, 7, 10, 13, 17, 20, 22, 24, 25, 25, 24, 22, 20, 17, 13, 10, 7, 5, 3, 2, 1]
        #     renewable_rescale = 0.1
        #     renewable_boundary = renewable_rescale * torch.tensor(renewable_avail)  # shape [24]

        #     u_b = u_batch.clone()

        #     # Reshape renewable_boundary to [1, 24, 1, 1] for broadcasting
        #     renewable_boundary = renewable_boundary.view(1, 24, 1, 1)

        #     # Subtract renewable_boundary from u_b
        #     u_delta_renewable = u_b - renewable_boundary

        #     u_surplus = torch.relu(u_delta_renewable)
            

        # upper bound on temperature loss
        if self.alpha_xh is None:
            loss_xh = 0
        else:
            loss_xh = self.alpha_xh * self.f_upper_bound_x(x_batch)       # shape = (S, 1, 1)

        # lower bound on temperature loss
        if self.alpha_xl is None:
            loss_xl = 0
        else:
            loss_xl = self.alpha_xl * self.f_lower_bound_x(x_batch) # shape = (S, 1, 1)

        self.l_xl = torch.sum(loss_xl, 0)/xs.shape[0]
        
        self.l_xh = torch.sum(loss_xh, 0)/xs.shape[0]

        # loss on switching pump ON/OFF, continuous or discrete or off
        switch_loss_mode = "off"
        if switch_loss_mode == "off":
            switch_loss = 0
        elif switch_loss_mode == "continuous" | "discrete":
            u_bool = u_batch >= 0.01
            switch_number = (u_bool[:, 1:, :, :] ^ u_bool[:, :-1, :, :]).float()
            if switch_loss_mode == "discrete":
                #discrete
                switch_penalty = switch_number
            if switch_loss_mode == "continuous":
                u_smooth = torch.sigmoid(u_batch)  # Output between 0 and 1
                # Compute the difference between consecutive time steps along the time dimension (dim=1)
                # This will give us the "change" between consecutive time steps
                # switch_penalty = torch.abs(u_smooth[:, 1:, :, :] - u_smooth[:, :-1, :, :])  # L1 difference
                # Optionally, use the L2 difference (squared difference) instead
                switch_penalty = (u_smooth[:, 1:, :, :] - u_smooth[:, :-1, :, :]).pow(2)  # L2 difference
            switch_switch_loss = switch_penalty.sum()
            alpha_switch = 0.07
            switch_loss = alpha_switch * switch_loss
            self.l_switch = torch.sum(switch_number.sum(), 0)

        # sum up all losses
        loss_val = loss_u + loss_xh + loss_xl           # + switch_loss          # shape = (S, 1, 1)

        # bound
        if self.sat_bound is not None:
            loss_val = torch.tanh(loss_val/self.sat_bound)  # shape = (*batch_dim, 1, 1)
        if self.loss_bound is not None:
            loss_val = self.loss_bound * loss_val           # shape = ((*batch_dim, 1, 1
        
        loss_val = torch.sum(loss_val, 0)/xs.shape[0]       # shape = (1, 1)
        
        return loss_val
    
    def f_switching(self, u):
        
        pass

    
    def f_upper_bound_x(self, x_batch): 
        """
        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
        """

        
        delta = x_batch - self.xmax

        loss_bound = torch.relu(delta) ##
        # loss_bound = torch.nn.functional.softplus(delta, beta = 2.0) ##
        loss_xh = loss_bound.sum(1)/loss_bound.shape[1]
        return loss_xh.reshape(-1,1,1)



    def f_lower_bound_x(self, x_batch,s = True):

        delta = self.xmin - x_batch  

        loss_bound = torch.relu(delta) ##
        # loss_bound = torch.nn.functional.softplus(delta, beta = 2.0)
        if s == True: 
            loss_xl = loss_bound.sum(1)/loss_bound.shape[1]
            return loss_xl.reshape(-1,1,1)
        else: 
            return loss_bound

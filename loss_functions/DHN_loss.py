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
        self, R, u_min, u_max, x_min,x_max,
        alpha_xl=None, alpha_xh=None
    ):
        
        self.umin = to_tensor(u_min).to(device)
        self.umax = to_tensor(u_max).to(device)
        self.xmin = to_tensor(x_min).to(device)
        self.xmax = to_tensor(x_max).to(device)

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
        self.tariff = torch.tensor(high+low).to(device)



    def forward(self, xs, us,dxref):
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
        dxref = dxref.reshape(*dxref.shape, 1)


        u2_batch = us.reshape(*us.shape, 1)
        # loss states = 1/T sum_{t=1}^T (x_t-xbar)^T Q (x_t-xbar)
        
        # loss control actions = 1/T sum_{t=1}^T u_t^T R u_t
        uTRu = self.R * torch.matmul(
            u_batch.transpose(-1, -2),
            u_batch
        )   # shape = (S, T, 1, 1)

        u_b = u_batch.clone()

        for i in range(13):
            u_b[:,i,:,:] = u_b[:,i,:,:]*10


        #loss_u = torch.sum(u_b, 1) / x_batch.shape[1]    # average over the time horizon. shape = (S, 1, 1)
        loss_u = torch.sum(uTRu, 1) / x_batch.shape[1] 

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

        # sum up all losses
        loss_val = loss_u + loss_xh + loss_xl          # shape = (S, 1, 1)
        
        loss_val = torch.sum(loss_val, 0)/xs.shape[0]       # shape = (1, 1)
        return loss_val 

    
    def f_upper_bound_x(self, x_batch): 
        """
        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
        """

        
        delta = x_batch - self.xmax


        loss_bound = torch.relu(delta)
        loss_xh = loss_bound.sum(1)/loss_bound.shape[1]
        return loss_xh.reshape(-1,1,1)



    def f_lower_bound_x(self, x_batch,s = True):

        delta = self.xmin - x_batch  


        loss_bound = torch.relu(delta)
        if s == True: 
            loss_xl = loss_bound.sum(1)/loss_bound.shape[1]
            return loss_xl.reshape(-1,1,1)
        else: 
            return loss_bound

import torch
from assistive_functions import to_tensor, saturate
import numpy as np
import torch.nn.functional as F
from config import device
from scipy.signal import place_poles


# ---------- SYSTEM ----------
class DHNSystem(torch.nn.Module):
    def __init__(self,mass,cop,cp,umax,umin,xref,gamma = 0.99,u_init = None):
        
        super().__init__()

        self.mass = mass
        self.cop = cop
        self.cp = cp
        self.umax = umax
        self.umin = umin
        self.xref = xref
        A = np.array([[gamma]])
        B = np.array([[self.cop/(self.mass*self.cp)]])

        desired_poles = [0.80]

        # Calculate the gain matrix K
        result = place_poles(A, B, desired_poles)
        K = result.gain_matrix



        self.A, self.B,self.K = to_tensor(A), to_tensor(B), to_tensor(K)

        self.I = torch.eye(self.A.shape[0]).to(device)

        self.B_inv = torch.inverse(self.B).to(device)

        self.Kr = self.K+F.linear(self.B_inv,self.I-self.A).to(device)

        u_i = F.linear(self.Kr,self.xref)

        self.x_init = to_tensor(np.array([[0]]))

        # Dimensions
        self.state_dim = self.A.shape[0]
        self.in_dim = self.B.shape[1]
        # Check matrices
        assert self.A.shape == (self.state_dim, self.state_dim)
        assert self.B.shape == (self.state_dim, self.in_dim)
        assert self.K.shape == (self.in_dim, self.state_dim)
        assert self.x_init.shape == (self.state_dim, 1)



        self.u_init = torch.full((1, int(self.x_init.shape[1])),float(0)).to(device) if u_init is None else u_init.reshape(1, -1)   # shape = (1, in_dim)
        #self.u_init = torch.full((1, int(self.x_init.shape[1])),-xref.item()).to(device) if u_init is None else u_init.reshape(1, -1)   # shape = (1, in_dim)

    def base_controller(self,x: torch.Tensor, u: torch.Tensor,I,Kr):

        u_cont = F.linear(x,-self.K) + F.linear(u,Kr) 
        
        
        return u_cont

    def noiseless_forward(self, x: torch.Tensor, u: torch.Tensor):
        x = x.view(-1, 1, self.state_dim)
        #delta x_ref
        dxref = u.view(-1, 1, self.in_dim)
        xref = self.xref.view(-1, 1, self.in_dim)
        
        u_base = self.base_controller(x,xref,self.I,self.Kr) 

        u_PB = F.linear(dxref,self.Kr)    


        u_cont = u_base + u_PB

        #Saturate u
        u_cont = torch.where(u_cont<0,0,torch.where(u_cont>self.umax,self.umax,u_cont))

        f = F.linear(x,self.A)+ F.linear(u_cont,self.B)

        self.u_cont = u_cont

        return f

    
    
    def forward(self, x, u, w):
        """
        forward of the plant with the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            next state.
        """

        return self.noiseless_forward(x, u) + w.view(-1, 1, self.state_dim)


    # simulation
    def rollout(self, controller, data: torch.Tensor):
        """
        rollout with state-feedback controller

        Args:
            - controller: state-feedback controller
            - data (torch.Tensor): batch of disturbance samples, with shape (batch_size, T, state_dim)
        """

        controller.reset()

        xs = (data[:, 0:1, :]/(self.mass*self.cp))

        dxref = controller.forward(xs[:, 0:1, :])
        us = torch.full(dxref.shape,self.u_cont.item()).to(device)

        for t in range(1, data.shape[1]):
            xs = torch.cat(
                (
                    xs,
                    self.forward(xs[:, t-1:t, :],dxref[:, t-1:t, :],data[:, t:t+1, :]/(self.mass*self.cp))
                    ),
                1
            )

            dxref = torch.cat(
                (dxref, controller.forward(xs[:, t:t+1, :])),
                1
            )
            us = torch.cat(
                (us, self.u_cont),
                1
            )
            
        controller.reset()
        
        return xs, us, dxref
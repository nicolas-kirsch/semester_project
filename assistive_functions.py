import torch
import numpy as np
from config import device


def to_tensor(x):
    return torch.from_numpy(x).contiguous().float().to(device) if isinstance(x, np.ndarray) else x


def saturate(u,umin,umax):
    u[u < umin] = umin
    u[u > umax] = umax

    return u.to(device)



def to_range(u,umin):
    u[u != 0] = u + umin

    return u.to(device)


"""def heaviside(u,k=12):
    heavi_u = torch.where(u<0,(torch.exp(k*u))/(1+torch.exp(k*u)),1/(1+torch.exp(-k*u))) 

    return heavi_u"""

def heaviside(u,m =10**(-3), alpha = 1):
    heavi_u = torch.minimum(torch.maximum(u/m+alpha/2-4,torch.zeros(u.shape).to(device)),alpha*torch.ones(u.shape).to(device))
    #heavi_u = torch.minimum(torch.maximum(u/m+alpha/2-4,0.0001*u),1+ alpha*u*0.0001)
    return heavi_u

class WrapLogger():
    def __init__(self, logger, verbose=True):
        self.can_log = (logger is not None)
        self.logger = logger
        self.verbose = verbose

    def info(self, msg):
        if self.can_log:
            self.logger.info(msg)
        if self.verbose:
            print(msg)

    def close(self):
        if not self.can_log:
            return
        while len(self.logger.handlers):
            h = self.logger.handlers[0]
            h.close()
            self.logger.removeHandler(h)

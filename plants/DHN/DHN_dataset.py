import torch
import numpy as np # TODO: move to torch
from plants import CostumDataset
import matplotlib.pyplot as plt
from config import device


class DHNDataset(CostumDataset):
    def __init__(self, random_seed, horizon, disturbance, state_dim,cp,mass,xmin= None,xmax=None,umin=None,umax=None):
        # experiment and file names
        exp_name = 'DHN'
        file_name = disturbance['type'].replace(" ", "_")+'_data_T'+str(horizon)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.cp = cp
        self.mass = mass
        self.horizon = horizon
        self.disturbance = disturbance
        self.state_dim = state_dim

        if umin == None:
            self.umin = torch.tensor([umin
                                  ]).to(device)
        else: 
            self.umin = umin

        if umax == None:
            self.umax = torch.tensor([umax
                                  ]).to(device)
        else: 
            self.umax = umax
            
        if xmin == None:
            self.xmin = torch.tensor([xmin
                                  ]).to(device)
        else: 
            self.xmin = xmin
            
        if xmax == None:
            self.xmax = torch.tensor([xmax
                                  ]).to(device)
        else: 
            self.xmax = xmax 



    # ---- data generation ----
    def _generate_data(self, num_samples, mode):
    
        # generate data
        n_data_total = num_samples
        n_states = 1
        n_w = n_states

        # Define the window size for the moving average
        window_size = 3

        rescale = 0.2  # use this to keep same profile but change demand load / original 0.2
        heat_demand =  [30,20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 90, 80, 70, 60, 50, 60, 80, 100, 90, 80, 70, 50, 40]      # OG

        # concatinated_heat_demand_48h = heat_demand + heat_demand #+ heat_demand
        # heat_demand = concatinated_heat_demand_48h

        train_noise = 3 ##
        test_noise = 3

        if mode == "train" :         ##
            noise_gain = train_noise
            pass
        elif mode == "test": 
            noise_gain = test_noise   
            # rescale = 0.2   # use this to keep same profile but change demand load / original 0.2
            # heat_demand = [100, 90, 80, 70, 60, 50, 30, 20, 25, 30, 35, 40, 50, 60, 70, 80, 60, 80, 90, 70, 50, 40, 100, 90]
            # heat_demand = heat_demand
            pass        # to have train and test with the same distribution

        else:
            raise NameError('Wrong data generation mode')

        # Compute the moving average        ##
        # smoothed_heat_demand = -np.convolve(heat_demand, np.ones(window_size)/window_size, mode='same')


        ## flat demand
        # flat_heat_demand =  [30, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 90, 80, 70, 60, 50, 55, 55, 55, 55, 55, 55, 55, 55]
        # flat_heat_demand += flat_heat_demand
        flat_heat_demand =  [30, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

        pad_width = window_size // 2
        flat_padded = np.pad(flat_heat_demand, pad_width, mode='edge')
        smoothed_heat_demand = -np.convolve(flat_padded, np.ones(window_size) / window_size, mode='valid')
        ##

        #Initial temperature (randomized)
        data_x0 = ((self.xmin + (self.xmax-self.xmin)*torch.rand(n_data_total, n_states).to(device)))

        #Generate consumption data (+ adding noise to it)
        d = torch.zeros(n_data_total,self.horizon,n_w)  
        for i in range(n_data_total):
            d[i] = (torch.from_numpy(smoothed_heat_demand).reshape(self.horizon,n_w) + noise_gain*torch.randn(self.horizon,n_w))*rescale ## was 3*


            d[i][0] = data_x0[i]*self.cp*self.mass

        #     plt.plot(list(range(len(d[i]))),d[i].cpu())
        # plt.title(("heat demand"))

        # plt.savefig("Toy_Data.png")  
        
        d

        return d
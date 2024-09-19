#!/usr/bin/env python3
import sys, os, logging, torch,time
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)


from config import device
from controllers import PerfBoostController
from arg_parser import argument_parser
from plants import DHNDataset, DHNSystem
from assistive_functions import WrapLogger
from loss_functions import DHNLoss


# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'DHN', 'saved_results',"log")
save_folder = os.path.join(save_path, 'perf_boost_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ----- parse and set experiment arguments -----
args = argument_parser()
# msg = print_args(args)    # TODO
# logger.info(msg)

mass = 2200 #kg
cop = 3.53 
cp = 1.16*10**(-3) #(kWh.J^-1.K^-1)

x0 = torch.tensor(float(15)).to(device)
dx_ref = torch.tensor(float(-5)).to(device)

T_min = torch.Tensor([28]).to(device)
T_max = torch.Tensor([38]).to(device)

T_ext = torch.Tensor([25]).to(device)
T_ref = torch.Tensor([33]).to(device)

x_min = T_min - T_ext 
x_max = T_max - T_ext
x_ref = T_ref - T_ext



dxref_min = x_min - x_ref 
dxref_max = x_max - x_ref

u_min = torch.Tensor([0]).to(device)
u_max = torch.Tensor([12.8]).to(device)

gamma = 0.99

# ------------ 1. Dataset ------------
disturbance = {
    'type':'normal noise',
}

dataset = DHNDataset(
    random_seed=args.random_seed, horizon=args.horizon,
    state_dim=args.state_dim, disturbance=disturbance,
    cp = cp, mass = mass,xmin=x_min,xmax=x_max,umin = u_min,umax=u_max
)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=20)
train_data, test_data = train_data.to(device), test_data.to(device)


max_train = torch.max(train_data)
min_train = torch.min(train_data)

#train_data = (train_data-min_train)/(max_train-min_train)

# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 1. Dataset ------------

sys = DHNSystem(
    mass=mass,cop = cop,gamma=gamma,cp = cp,umin = u_min,umax = u_max,xref=x_ref
).to(device)


ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init, output_init=sys.u_init,dmin=min_train,dmax = max_train,
    dim_internal=args.dim_internal, dim_nl=args.l,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)


# ------------ 4. Loss ------------
#Size of the minimization 
loss_fn = DHNLoss(
    R=args.alpha_u, u_min=u_min, u_max=u_max, x_min=x_min,x_max=x_max,
    alpha_xh=50,    
    alpha_xl=50,
)


# ------------ 5. Optimizer ------------
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
valid_data = train_data      # use the entire train data for validation




# ------------ 6. Training ------------
logger.info('\n------------ Begin training ------------')
best_valid_loss = 1e6
t = time.time()
for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()



        # simulate over horizon steps
        x_log, u_log, dxref = sys.rollout(controller=ctl, data=train_data_batch)

        # loss of this rollout
        loss = loss_fn.forward(x_log, u_log,dxref)

        loss.backward()
        optimizer.step()





    # print info
    if epoch%args.log_epoch == 0:
        msg = 'Epoch: %i --- train loss: %.2f --- Loss xl : %.2f'% (epoch, loss, loss_fn.l_xl.item())
        #msg +='--- Loss xh : %.2f ---  loss ul: %.2f---  loss uh: %.2f'% (loss_xh, loss_ul, loss_uh)
        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                x_log_valid, u_log_valid, dxref_valid= sys.rollout(
                    controller=ctl, data=valid_data
                )
                if epoch == 0: 

                    x_log_test = x_log_valid.cpu()
                    u_log_test = u_log_valid.cpu()


                # loss of the valid data
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid,dxref_valid)

            msg += ' ---||--- validation loss: %.2f  --- Loss x low: %.2f ---  loss x high: %.2f' % (
                loss_valid,loss_fn.l_xl.item(),loss_fn.l_xh.item())
            # compare with the best valid loss
            if loss_valid.item()<best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_params = ctl.get_parameters_as_vector()  # record state dict if best on valid
                msg += ' (best so far)'
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % (duration)
        logger.info(msg)
        t = time.time()

# set to best seen during training
if args.return_best:
    ctl.set_parameters_as_vector(best_params)


with torch.no_grad():
    x_log_test, u_log_test, dxref_test = sys.rollout(
        controller=ctl, data=test_data
    )


x_log_test = x_log_test.cpu()
u_log_test = u_log_test.cpu()
dxref_test = dxref_test.cpu()



plt.figure()
for i in range(test_data.shape[0]): 
    plt.plot(range(test_data.shape[1]),x_log_test[i]+T_ext.cpu())
    plt.plot(range(test_data.shape[1]),[T_min.cpu()]*(test_data.shape[1]), "--", c = "grey")
    plt.plot(range(test_data.shape[1]),[T_max.cpu()]*(test_data.shape[1]), "--",c = "grey" )
    plt.title("X profile over the horizon")
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature (°C)")
plt.savefig("saved_results/x_log.png")

plt.figure()
for i in range(test_data.shape[0]): 
    plt.plot(range(test_data.shape[1]),dxref_test[i])
    plt.title("DXref profile over the horizon")
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature (°C)")
plt.savefig("saved_results/dxref_log.png")


plt.figure()
for i in range(test_data.shape[0]):     
    plt.plot(range(test_data.shape[1]),u_log_test[i])
    plt.plot(range(test_data.shape[1]),[u_min.cpu()]*(test_data.shape[1]), "--", c = "grey")
    plt.plot(range(test_data.shape[1]),[u_max.cpu()]*(test_data.shape[1]), "--",c = "grey" )
    plt.title("U profile over the horizon")
    plt.xlabel("Time (h)")
    plt.ylabel("Energy (kWh)")
plt.savefig("saved_results/u_profile.png")


plt.show()
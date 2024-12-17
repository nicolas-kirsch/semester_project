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
from controllers import  PerfBoostController #
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

peak_tarif = False 

mass = 2200 #kg
cop = 3.53 
cp = 1.16*10**(-3) #(kWh.J^-1.K^-1)

x0 = torch.tensor(float(15)).to(device)
dx_ref = torch.tensor(float(-5)).to(device)

T_min = torch.Tensor([28]).to(device)
T_max = torch.Tensor([38]).to(device)

T_ext = torch.Tensor([25]).to(device)
T_ref = torch.Tensor([33]).to(device)           

# changing into correct states to control
x_min = T_min - T_ext 
x_max = T_max - T_ext
x_ref = T_ref - T_ext

dxref_min = x_min - x_ref 
dxref_max = x_max - x_ref

u_min = torch.Tensor([0]).to(device)
u_max = torch.Tensor([12.8]).to(device)




gamma = 0.99

# ------------ 1. Dataset ------------  
disturbance = {                        # useless, remenance of old code
    'type':'normal noise',
}

dataset = DHNDataset(
    random_seed=args.random_seed, horizon=args.horizon,
    state_dim=args.state_dim, disturbance=disturbance,
    cp = cp, mass = mass,xmin=x_min,xmax=x_max,umin = u_min,umax=u_max
)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=1000)      ## was 20
_ , test_data_plot = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=30)         # have a smaller testing dataset for plotting

train_data, test_data = train_data.to(device), test_data.to(device)

max_train = torch.max(train_data)
min_train = torch.min(train_data)

#train_data = (train_data-min_train)/(max_train-min_train)

# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.num_rollouts, shuffle=False) ## might be slower is batch size too big, shuffle was True, and batch_size was args.batch_size (or smthng)

# ------------ 1. Dataset ------------

sys = DHNSystem(
    mass=mass,cop = cop,gamma=gamma,cp = cp,umin = u_min,umax = u_max,xref=x_ref
).to(device)


ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init, output_init=sys.u_init,dmin=min_train,dmax = max_train,train_method="empirical",
    dim_internal=args.dim_internal, dim_nl=args.l,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)

print(" **** shape of total parameters: ",ctl.get_parameters_as_vector().shape)
print(ctl.c_ren.B2)

# ------------ 4. Loss ------------
#Size of the minimization 
loss_fn = DHNLoss(
    R=args.alpha_u, u_min=u_min, u_max=u_max, x_min=x_min,x_max=x_max,
    peak = peak_tarif,      # multiplies by 10 for first 13 hours
    alpha_xh=50,   
    alpha_xl=50,   
)

"""x_log_tenta, u_log_tenta, dxref_tenta = sys.rollout(
        controller=ctl, data=test_data
    )
print("ZAZAZAZAZAZ",dxref_tenta[0])"""

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
        loss = loss_fn.forward(x_log, u_log)

        loss.backward()
        optimizer.step()

    # print info
    if epoch%args.log_epoch == 0:
        print('---------------')
        msg = 'Epoch: %i --- TRAIN LOSS : %.2f --- Loss xl : %.2f, Loss xh : %.2f'% (epoch, loss, loss_fn.l_xl.item(), loss_fn.l_xh.item())
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
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid)

            msg += ' ---||---  VALIDATION LOSS: %.2f  --- Loss xl: %.2f ---  Loss xh: %.2f' % (
                loss_valid,loss_fn.l_xl.item(),loss_fn.l_xh.item())
            # compare with the best valid loss
            if loss_valid.item()<best_valid_loss:
                x_log_valid_best = x_log_valid          ##
                u_log_valid_best = u_log_valid
                dxref_valid_best = dxref_valid
                best_valid_loss = loss_valid.item()
                best_params = ctl.get_parameters_as_vector()  # record state dict if best on valid
                msg += ' (*** best so far ***)'
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
    test_loss = loss_fn.forward(x_log_test, u_log_test)[0][0].item()
    print(f"\n TEST loss: {test_loss:.2f}")  ##

    x_log_test_plot, u_log_test_plot, dxref_test_plot = sys.rollout(
        controller=ctl, data=test_data_plot
    )

x_log_test = x_log_test.cpu()
u_log_test = u_log_test.cpu()
dxref_test = dxref_test.cpu()

# plt.figure()
# for i in range(test_data.shape[0]): 
#     plt.plot(range(test_data.shape[1]),x_log_test[i]+T_ext.cpu())
#     plt.plot(range(test_data.shape[1]),[T_min.cpu()]*(test_data.shape[1]), "--", c = "grey")
#     plt.plot(range(test_data.shape[1]),[T_max.cpu()]*(test_data.shape[1]), "--",c = "grey" )
#     plt.title(f"X profile, training size 1024, validation loss: {best_valid_loss:.2f}, testing loss: {test_loss:.2f}")
#     plt.xlabel("Time (h)")
#     plt.ylabel("Temperature (°C)")
# plt.savefig("X evol training size 1024.png")

# plt.figure()
# for i in range(test_data.shape[0]): 
#     plt.plot(range(test_data.shape[1]),dxref_test[i])
#     plt.title("DXref profile over the horizon")
#     plt.xlabel("Time (h)")
#     plt.ylabel("Temperature (°C)")
# plt.savefig("saved_results/dxref_log.png")

# plt.figure()
# for i in range(test_data.shape[0]):     
#     plt.plot(range(test_data.shape[1]),u_log_test[i])
#     plt.plot(range(test_data.shape[1]),[u_min.cpu()]*(test_data.shape[1]), "--", c = "grey")
#     plt.plot(range(test_data.shape[1]),[u_max.cpu()]*(test_data.shape[1]), "--",c = "grey" )
#     plt.title("U profile over the horizon")
#     plt.xlabel("Time (h)")
#     plt.ylabel("Energy (kWh)")
# plt.savefig("saved_results/u_profile.png")

# plt.show()

plot_mode = "new"       # one if only plot train, two for testing and training datasets. "two_v2 is a failed attempt at plotting CI"

if plot_mode == "one":

    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    # Plot 1: X profile over the horizon
    for i in range(test_data.shape[0]): 
        axs[0, 0].plot(range(test_data.shape[1]), x_log_test[i] + T_ext.cpu())
        axs[0, 0].plot(range(test_data.shape[1]), [T_min.cpu()] * test_data.shape[1], "--", c="grey")
        axs[0, 0].plot(range(test_data.shape[1]), [T_max.cpu()] * test_data.shape[1], "--", c="grey")
    axs[0, 0].set_title("X profile over the horizon")
    axs[0, 0].set_xlabel("Time (h)")
    axs[0, 0].set_ylabel("Temperature (°C)")
    axs[0, 0].grid()


    # Plot 2: DXref profile over the horizon
    for i in range(test_data.shape[0]): 
        axs[0, 1].plot(range(test_data.shape[1]), dxref_test[i])
    axs[0, 1].set_title("DXref profile over the horizon")
    axs[0, 1].set_xlabel("Time (h)")
    axs[0, 1].set_ylabel("Temperature (°C)")
    axs[0, 1].grid()

    # Plot 3: U profile over the horizon
    for i in range(test_data.shape[0]): 
        axs[1, 1].plot(range(test_data.shape[1]), [u_min.cpu()] * test_data.shape[1], "--", c="grey")
        axs[1, 1].plot(range(test_data.shape[1]), [u_max.cpu()] * test_data.shape[1], "--", c="grey")
        axs[1, 1].plot(range(test_data.shape[1]), u_log_test[i])
    axs[1, 1].set_title("U profile over the horizon")
    axs[1, 1].set_xlabel("Time (h)")
    axs[1, 1].set_ylabel("Energy (kWh)")
    axs[1, 1].grid()

    # Plot 4: Test data profile
    for i in range(test_data.shape[0]):
        axs[1, 0].plot(range(test_data.shape[1]), test_data[i])
        # axs[1, 0].plot(range(test_data.shape[1]), -test_data[i])

    axs[1, 0].set_title("Test Data profile over the horizon")
    axs[1, 0].set_xlabel("Time (h)")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

    # plt.suptitle(f'System evolution with gamma = {gamma}, Tref = {T_ref[0].item():.0f}, no load, no PB control', fontsize=17)
    plt.suptitle(f'System evolution with heat load, base controller (Tref = {T_ref[0].item():.0f})', fontsize=17)

    plt.savefig(f'System evol, base controller, with demand .png')

    # plt.savefig("testing_no_load_no_PB.png")
    plt.show()

if plot_mode =="one_v2":        # only x and 
    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 1, figsize=(13, 9))

    # Plot 1: X profile over the horizon
    for i in range(test_data.shape[0]): 
        axs[0].plot(range(test_data.shape[1]), x_log_test[i] + T_ext.cpu())
        axs[0].plot(range(test_data.shape[1]), [T_min.cpu()] * test_data.shape[1], "--", c="grey")
        axs[0].plot(range(test_data.shape[1]), [T_max.cpu()] * test_data.shape[1], "--", c="grey")
    axs[0].set_title("X profile over the horizon")
    axs[0].set_xlabel("Time (h)")
    axs[0].set_ylabel("Temperature (°C)")
    axs[0].grid()
    axs[0].axvline(x = 24,  color = 'r', linestyle='dashed')


    # Plot 4: Test data profile
    for i in range(test_data.shape[0]):
        axs[1].plot(range(test_data.shape[1]), test_data[i])
        # axs[1, 0].plot(range(test_data.shape[1]), -test_data[i])

    axs[1].axvline(x = 24,  color = 'r', linestyle='dashed')
    axs[1].set_title("Test Data profile over the horizon")
    axs[1].set_xlabel("Time (h)")
    axs[1].set_ylabel("Value")
    axs[1].grid()
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

    # plt.suptitle(f'System evolution with gamma = {gamma}, Tref = {T_ref[0].item():.0f}, no load, no PB control', fontsize=17)
    plt.suptitle(f'System evolution with flat-ended heat demand 48h', fontsize=17)

    plt.savefig(f'System evol flat-ended heat demandm 48.png')

    # plt.savefig("testing_no_load_no_PB.png")
    plt.show()



###### combined plots version 2 train and test
if plot_mode == "two":
    from matplotlib.colors import Normalize

    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    # Colormaps for train and test data
    train_color = "black"
    test_color = "red"
    alpha_value = 0.9  # Transparency for overlapping effect # was 0.3

    # Normalize functions for color scaling independent within each subplot
    norm_test = Normalize(vmin=0, vmax=test_data.shape[0] - 1)
    norm_train = Normalize(vmin=0, vmax=train_data.shape[0] - 1)

    # Plot 1: X profile over the horizon
    for i in range(test_data.shape[0]):
        axs[0, 0].plot(range(test_data.shape[1]), x_log_test[i] + T_ext.cpu(), alpha=alpha_value, label=f"Test {i}")
        axs[0, 0].plot(range(test_data.shape[1]), [T_min.cpu()] * test_data.shape[1], "--", c="grey")
        axs[0, 0].plot(range(test_data.shape[1]), [T_max.cpu()] * test_data.shape[1], "--", c="grey")
    for i in range(x_log_valid_best.shape[0]):
        axs[0, 0].plot(range(x_log_valid_best.shape[1]), x_log_valid_best[i] + T_ext.cpu(), color = train_color, alpha=alpha_value, label=f"Train {i}")
    axs[0, 0].set_title("X profile over the horizon")
    axs[0, 0].set_xlabel("Time (h)")
    axs[0, 0].set_ylabel("Temperature (°C)")
    axs[0, 0].grid()
    # axs[0, 0].axvline(x = 24,  color = 'r', linestyle='dashed')


    # Plot 2: DXref profile over the horizon
    for i in range( dxref_test.shape[0]):  # Loop over test data rows
        axs[0, 1].plot(range(test_data.shape[1]), dxref_test[i], alpha=alpha_value, label=f"Test {i}")
    for i in range(dxref_valid_best.shape[0]):  # Loop over train data rows
        axs[0, 1].plot(range(dxref_valid_best.shape[1]), dxref_valid_best[i], color=train_color, alpha=alpha_value, label=f"Train {i}")
    axs[0, 1].set_title("DXref profile over the horizon")
    axs[0, 1].set_xlabel("Time (h)")
    axs[0, 1].set_ylabel("Temperature (°C)")
    axs[0, 1].grid()
    # axs[0, 1].axvline(x = 24,  color = 'r', linestyle='dashed')


    # Plot 3: U profile over the horizon
    for i in range(u_log_test.shape[0]):  # Loop over test data rows
        axs[1, 1].plot(range(test_data.shape[1]), [u_min.cpu()] * test_data.shape[1], "--", c="grey")
        axs[1, 1].plot(range(test_data.shape[1]), [u_max.cpu()] * test_data.shape[1], "--", c="grey")
        axs[1, 1].plot(range(test_data.shape[1]), u_log_test[i], alpha=alpha_value, label=f"Test {i}")

    for i in range(u_log_valid_best.shape[0]):  # Loop over train data rows
        axs[1, 1].plot(range(u_log_valid_best.shape[1]), u_log_valid_best[i], color=train_color, alpha=alpha_value, label=f"Train {i}")
    axs[1, 1].set_title("U profile over the horizon")
    axs[1, 1].set_xlabel("Time (h)")
    axs[1, 1].set_ylabel("Energy (kWh)")
    axs[1, 1].grid()
    # axs[1, 1].axvline(x = 24,  color = 'r', linestyle='dashed')


    # Plot 4: Test and Train Data profile
    for i in range(test_data.shape[0]):
        axs[1, 0].plot(range(test_data.shape[1]), test_data[i], alpha=alpha_value, label=f"Test {i}")
    for i in range(train_data.shape[0]):
        axs[1, 0].plot(range(train_data.shape[1]), train_data[i], color=train_color, alpha=alpha_value, label=f"Train {i}")
    axs[1, 0].set_title("Test and Train Data profile over the horizon")
    axs[1, 0].set_xlabel("Time (h)")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].set_ylim([-20,0])
    axs[1, 0].grid()
    # axs[1, 0].axvline(x = 24,  color = 'r', linestyle='dashed')


    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

    plt.suptitle(f'System evolution (validation loss: {best_valid_loss:.2f}, test loss: {test_loss:.2f})')

    plt.savefig("System evolution.png")
    # Show the figure
    plt.show()



if plot_mode == "two_v2":

    # Updated compute_confidence_interval to work with PyTorch tensors
    def compute_confidence_interval(data, confidence=0.95):
        mean = data.mean(dim=0)
        error_margin = 1.96 * torch.std(data, dim=0) / torch.sqrt(torch.tensor(data.shape[0], dtype=torch.float32))
        lower_bound = mean - error_margin
        upper_bound = mean + error_margin
        return mean, lower_bound, upper_bound

    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    # Colors for train and test data
    train_color = "blue"
    test_color = "red"
    alpha_value = 0.3  # Transparency for the confidence interval fill

    # Plot 1: X profile over the horizon
    mean_test, lower_test, upper_test = compute_confidence_interval(x_log_test + T_ext.cpu())
    mean_train, lower_train, upper_train = compute_confidence_interval(x_log_valid_best + T_ext.cpu())

    axs[0, 0].plot(range(mean_test.shape[0]), mean_test.numpy(), color=test_color, label="Test Mean")
    axs[0, 0].fill_between(range(mean_test.shape[0]), lower_test.squeeze().numpy(), upper_test.squeeze().numpy(), color=test_color, alpha=alpha_value)
    
    axs[0, 0].plot(range(mean_train.shape[0]), mean_train.numpy(), color=train_color, label="Train Mean")
    axs[0, 0].fill_between(range(mean_train.shape[0]), lower_train.squeeze().numpy(), upper_train.squeeze().numpy(), color=train_color, alpha=alpha_value)
    
    axs[0, 0].plot(range(mean_test.shape[0]), [T_min.cpu()] * mean_test.shape[0], "--", c="grey")
    axs[0, 0].plot(range(mean_test.shape[0]), [T_max.cpu()] * mean_test.shape[0], "--", c="grey")
    
    axs[0, 0].set_title("X profile over the horizon")
    axs[0, 0].set_xlabel("Time (h)")
    axs[0, 0].set_ylabel("Temperature (°C)")
    axs[0, 0].grid()
    axs[0, 0].legend(loc='upper right')  # Add legend for this subplot

    # Plot 2: DXref profile over the horizon
    mean_test_dxref, lower_test_dxref, upper_test_dxref = compute_confidence_interval(dxref_test)
    mean_train_dxref, lower_train_dxref, upper_train_dxref = compute_confidence_interval(dxref_valid_best)

    axs[0, 1].plot(range(mean_test_dxref.shape[0]), mean_test_dxref.numpy(), color=test_color, label="Test Mean")
    axs[0, 1].fill_between(range(mean_test_dxref.shape[0]), lower_test_dxref.squeeze().numpy(), upper_test_dxref.squeeze().numpy(), color=test_color, alpha=alpha_value)
    
    axs[0, 1].plot(range(mean_train_dxref.shape[0]), mean_train_dxref.numpy(), color=train_color, label="Train Mean")
    axs[0, 1].fill_between(range(mean_train_dxref.shape[0]), lower_train_dxref.squeeze().numpy(), upper_train_dxref.squeeze().numpy(), color=train_color, alpha=alpha_value)
    
    axs[0, 1].set_title("DXref profile over the horizon")
    axs[0, 1].set_xlabel("Time (h)")
    axs[0, 1].set_ylabel("Temperature (°C)")
    axs[0, 1].grid()
    axs[0, 1].legend(loc='upper right')  # Add legend for this subplot

    # Plot 3: U profile over the horizon
    mean_test_u, lower_test_u, upper_test_u = compute_confidence_interval(u_log_test)
    mean_train_u, lower_train_u, upper_train_u = compute_confidence_interval(u_log_valid_best)

    axs[1, 1].plot(range(mean_test_u.shape[0]), mean_test_u.numpy(), color=test_color, label="Test Mean")
    axs[1, 1].fill_between(range(mean_test_u.shape[0]), lower_test_u.squeeze().numpy(), upper_test_u.squeeze().numpy(), color=test_color, alpha=alpha_value)
    
    axs[1, 1].plot(range(mean_train_u.shape[0]), mean_train_u.numpy(), color=train_color, label="Train Mean")
    axs[1, 1].fill_between(range(mean_train_u.shape[0]), lower_train_u.squeeze().numpy(), upper_train_u.squeeze().numpy(), color=train_color, alpha=alpha_value)
    
    axs[1, 1].plot(range(mean_test_u.shape[0]), [u_min.cpu()] * mean_test_u.shape[0], "--", c="grey")
    axs[1, 1].plot(range(mean_test_u.shape[0]), [u_max.cpu()] * mean_test_u.shape[0], "--", c="grey")
    
    axs[1, 1].set_title("U profile over the horizon")
    axs[1, 1].set_xlabel("Time (h)")
    axs[1, 1].set_ylabel("Energy (kWh)")
    axs[1, 1].grid()
    axs[1, 1].legend(loc='upper right')  # Add legend for this subplot

    # Plot 4: Test and Train Data profile over the horizon
    mean_test_data, lower_test_data, upper_test_data = compute_confidence_interval(test_data)
    mean_train_data, lower_train_data, upper_train_data = compute_confidence_interval(train_data)

    axs[1, 0].plot(range(mean_test_data.shape[0]), mean_test_data.numpy(), color=test_color, label="Test Mean")
    axs[1, 0].fill_between(range(mean_test_data.shape[0]), lower_test_data.squeeze().numpy(), upper_test_data.squeeze().numpy(), color=test_color, alpha=alpha_value)
    
    axs[1, 0].plot(range(mean_train_data.shape[0]), mean_train_data.numpy(), color=train_color, label="Train Mean")
    axs[1, 0].fill_between(range(mean_train_data.shape[0]), lower_train_data.squeeze().numpy(), upper_train_data.squeeze().numpy(), color=train_color, alpha=alpha_value)
    
    axs[1, 0].set_title("Test and Train Data profile over the horizon")
    axs[1, 0].set_xlabel("Time (h)")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].grid()
    axs[1, 0].legend(loc='upper right')  # Add legend for this subplot

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f'System evolution (validation loss: {best_valid_loss:.2f}, test loss: {test_loss:.2f})')
    
    # Save and show the figure
    plt.savefig("Evolution_system.png")
    plt.show()

###### combined plots version 2 train and test
if plot_mode == "new":
    from matplotlib.colors import Normalize

    train_marker = 'o'

    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    # Colormaps for train and test data
    train_color = "black"
    test_color = None
    alpha_value = 1.0  # was 0.3 or 0.9

    # Normalize functions for color scaling independent within each subplot
    norm_test = Normalize(vmin=0, vmax=test_data_plot.shape[0] - 1)
    norm_train = Normalize(vmin=0, vmax=train_data.shape[0] - 1)

    # Plot 1: X profile over the horizon
    for i in range(test_data_plot.shape[0]):
        axs[0, 0].plot(range(test_data_plot.shape[1]), x_log_test_plot[i] + T_ext.cpu(), alpha=alpha_value, label=f"Test {i}", color = test_color)
        axs[0, 0].plot(range(test_data_plot.shape[1]), [T_min.cpu()] * test_data_plot.shape[1], "--", color = "grey")
        axs[0, 0].plot(range(test_data_plot.shape[1]), [T_max.cpu()] * test_data_plot.shape[1], "--", color = "grey")
    for i in range(x_log_valid_best.shape[0]):
        axs[0, 0].plot(range(x_log_valid_best.shape[1]), x_log_valid_best[i] + T_ext.cpu(), color = train_color, alpha=alpha_value, marker=train_marker, label=f"Train {i}")
    axs[0, 0].set_title("X profile over the horizon")
    axs[0, 0].set_xlabel("Time (h)")
    axs[0, 0].set_ylabel("Temperature (°C)")
    axs[0, 0].grid()
    # axs[0, 0].axvline(x = 24,  color = 'r', linestyle='dashed')


    # Plot 2: DXref profile over the horizon
    for i in range( dxref_test_plot.shape[0]):  # Loop over test data rows
        axs[0, 1].plot(range(test_data_plot.shape[1]), dxref_test_plot[i], alpha=alpha_value, color = test_color, label=f"Test {i}")
    for i in range(dxref_valid_best.shape[0]):  # Loop over train data rows
        axs[0, 1].plot(range(dxref_valid_best.shape[1]), dxref_valid_best[i], color=train_color, alpha=alpha_value,marker=train_marker, label=f"Train {i}")
    axs[0, 1].set_title("DXref profile over the horizon")
    axs[0, 1].set_xlabel("Time (h)")
    axs[0, 1].set_ylabel("Temperature (°C)")
    axs[0, 1].grid()
    # axs[0, 1].axvline(x = 24,  color = 'r', linestyle='dashed')


    # Plot 3: U profile over the horizon
    for i in range(u_log_test_plot.shape[0]):  # Loop over test data rows
        axs[1, 1].plot(range(test_data_plot.shape[1]), [u_min.cpu()] * test_data_plot.shape[1], "--", c="grey")
        axs[1, 1].plot(range(test_data_plot.shape[1]), [u_max.cpu()] * test_data_plot.shape[1], "--", c="grey")
        axs[1, 1].plot(range(test_data_plot.shape[1]), u_log_test_plot[i], alpha=alpha_value, color = test_color, label=f"Test {i}")

    for i in range(u_log_valid_best.shape[0]):  # Loop over train data rows
        axs[1, 1].plot(range(u_log_valid_best.shape[1]), u_log_valid_best[i], color=train_color, alpha=alpha_value, marker=train_marker, label=f"Train {i}")
    axs[1, 1].set_title("U profile over the horizon")
    axs[1, 1].set_xlabel("Time (h)")
    axs[1, 1].set_ylabel("Energy (kWh)")
    axs[1, 1].grid()
    # axs[1, 1].axvline(x = 24,  color = 'r', linestyle='dashed')


    # Plot 4: Test and Train Data profile
    for i in range(test_data_plot.shape[0]):
        axs[1, 0].plot(range(test_data_plot.shape[1]), test_data_plot[i], color = test_color, alpha=alpha_value, label=f"Test {i}")
    for i in range(train_data.shape[0]):
        axs[1, 0].plot(range(train_data.shape[1]), train_data[i], color=train_color, alpha=alpha_value, marker=train_marker, label=f"Train {i}")
    axs[1, 0].set_title("Test and Train Data profile over the horizon")
    axs[1, 0].set_xlabel("Time (h)")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].set_ylim([-20,0])
    axs[1, 0].grid()
    # axs[1, 0].axvline(x = 24,  color = 'r', linestyle='dashed')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

    plt.suptitle(f'System evolution (training loss: {best_valid_loss:.2f}, test loss: {test_loss:.2f})')

    plt.savefig("System evolution.png")
    # Show the figure
    plt.show()

import sys, os, logging, torch, math
from datetime import datetime
from torch.utils.data import DataLoader
from pyro.distributions import Normal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from config import device
    # from utils.plot_functions import *
# from plants import RobotsSystem, RobotsDataset
# from loss_functions import RobotsLossMultiBatch
from arg_parser import argument_parser, print_args
from inference_algs.distributions import GibbsPosterior
from controllers import PerfBoostController, SVGDCont
import matplotlib.pyplot as plt 
from plants import DHNDataset, DHNSystem
from assistive_functions import WrapLogger
from loss_functions import DHNLoss

"""

"""

TRAIN_METHOD = 'SVGD'

# -------- parameters for DHN systems ------------ #

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

# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args)

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
save_folder = os.path.join(save_path, args.cont_type+'_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('ren_controller_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

logger.info('---------- ' + TRAIN_METHOD + ' ----------\n\n')
logger.info(msg)
torch.manual_seed(args.random_seed)

# ------------ 1. Dataset ------------
# dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, n_agents=2)
dataset = DHNDataset(
    random_seed=args.random_seed, horizon=args.horizon,
    state_dim=args.state_dim, 
    cp = cp, mass = mass,xmin=x_min,xmax=x_max,umin = u_min,umax=u_max
)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=1000)
_ , test_data_plot = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=30)         # have a smaller testing dataset for plotting
train_data, test_data = train_data.to(device), test_data.to(device)
max_train = torch.max(train_data)
min_train = torch.min(train_data)
# data for plots
t_ext = args.horizon * 4
plot_data = torch.zeros(1, t_ext, train_data.shape[-1], device=device)
    # plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar) ##
plot_data = plot_data.to(device)
# batch the data
# train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) # TODO
train_dataloader = DataLoader(train_data, batch_size=args.num_rollouts, shuffle=False)

# ------------ 2. Plant ------------
    # plant_input_init = None     # all zero
    # plant_state_init = None     # same as xbar
    # sys = RobotsSystem(
    #     xbar=dataset.xbar, x_init=plant_state_init,
    #     u_init=plant_input_init, linear_plant=args.linearize_plant, k=args.spring_const
    # ).to(device)

sys = DHNSystem(
    mass=mass,cop = cop,gamma=gamma,cp = cp,umin = u_min,umax = u_max,xref=x_ref
).to(device)


# ------------ 3. Controller ------------

# ******* empirical controller *********

from controllers import old_PerfBoostController as old_PerfBoostController

empirical_ctl = old_PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init, output_init=sys.u_init, dmin=min_train,dmax = max_train,
    dim_internal=args.dim_internal, dim_nl=args.l,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)
# print("empirical params: ", empirical_ctl.get_parameters_as_vector())
print(" *** empirical params shape: ", empirical_ctl.get_parameters_as_vector().shape)

# *************

ctl_generic = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init, output_init=sys.u_init,
    dim_internal=args.dim_internal, dim_nl=args.l,
    initialization_std=args.cont_init_std,
    output_amplification=20, train_method=TRAIN_METHOD
).to(device)

print(" *** SVGD params shape: ", ctl_generic.get_parameters_as_vector().shape)

    # elif args.cont_type=='Affine':
    #     ctl_generic = AffineController(
    #         weight=torch.zeros(sys.in_dim, sys.state_dim, device=device, dtype=torch.float32),
    #         bias=torch.zeros(sys.in_dim, 1, device=device, dtype=torch.float32),
    #         train_method=TRAIN_METHOD
    #     )
    # elif args.cont_type=='NN':
    #     ctl_generic = NNController(
    #         in_dim=sys.state_dim, out_dim=sys.in_dim, layer_sizes=args.layer_sizes,
    #         train_method=TRAIN_METHOD
    #     )
    # else:
    #     raise KeyError('[Err] args.cont_type must be PerfBoost, NN, or Affine.')

num_params = ctl_generic.num_params
# logger.info('[INFO] Controller is of type ' + args.cont_type + ' and has %i parameters.' % num_params)
logger.info('[INFO] Controller is of type Perf Booster and has %i parameters.' % num_params)

# ------------ 4. Loss ------------
    # Q = torch.kron(torch.eye(args.n_agents), torch.eye(4)).to(device)   # TODO: move to args and print info
    # loss_bound = 1
    # x0 = dataset.x0.reshape(1, -1).to(device)
    # sat_bound = torch.matmul(torch.matmul(x0, Q), x0.t())
    # sat_bound += 0 if args.alpha_col is None else args.alpha_col
    # sat_bound += 0 if args.alpha_obst is None else args.alpha_obst
    # sat_bound = sat_bound/20
    # logger.info('Loss saturates at: '+str(sat_bound))
    # bounded_loss_fn = RobotsLossMultiBatch(
    #     Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
    #     loss_bound=loss_bound, sat_bound=sat_bound.to(device),
    #     alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    #     min_dist=args.min_dist if args.col_av else None,
    #     n_agents=sys.n_agents if args.col_av else None,
    # )
    # original_loss_fn = RobotsLossMultiBatch(
    #     Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
    #     loss_bound=None, sat_bound=None,
    #     alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    #     min_dist=args.min_dist if args.col_av else None,
    #     n_agents=sys.n_agents if args.col_av else None,
    # )

loss_bound = 1

sat_bound = torch.tensor([[250]])       ## random

bounded_loss_fn = DHNLoss(
    R=args.alpha_u, u_min=u_min, u_max=u_max, x_min=x_min,x_max=x_max,
    peak = peak_tarif,
    alpha_xh=50,   
    alpha_xl=50,   
    loss_bound=loss_bound, sat_bound=sat_bound.to(device),
)


original_loss_fn = DHNLoss(
    R=args.alpha_u, u_min=u_min, u_max=u_max, x_min=x_min,x_max=x_max,
    peak = peak_tarif,
    alpha_xh=50,   
    alpha_xl=50,   
    loss_bound=None, sat_bound=None,
)

# ------------ 5. Prior ------------
if args.cont_type in ['Affine', 'NN']:
    prior_dict = {
        'type':'Gaussian', 'type_w':'Gaussian',
        'type_b':'Gaussian_biased',
        'weight_loc':0, 'weight_scale':1,
        'bias_loc':0, 'bias_scale':5,
    }
else:       # default: PerfBooster
    prior_std = 7
    prior_dict = {'type':'Gaussian'}
    training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
    for name in training_param_names:
        prior_dict[name+'_loc'] = 0
        prior_dict[name+'_scale'] = prior_std

# ------------ 6. Posterior ------------
gibbs_lambda_star = (8*args.num_rollouts*math.log(1/args.delta))**0.5   # lambda for Gibbs, missing /C

if args.gibbs_lambda is None:
    gibbs_lambda = gibbs_lambda_star
else:
    gibbs_lambda = args.gibbs_lambda

logger.info('gibbs_lambda: %.2f' % gibbs_lambda + ' (use lambda_*)' if gibbs_lambda == gibbs_lambda_star else '')
# define target distribution
gibbs_posteior = GibbsPosterior(
    loss_fn=bounded_loss_fn, lambda_=gibbs_lambda, prior_dict=prior_dict,
    # attributes of the CL system
    controller=ctl_generic, sys=sys,
    # misc
    logger=logger,
)

# ****** INIT SVGD ******

num_particles = 1
# lr = 1e-2
early_stopping = True
# initialize trainable params
# initialization_std = 0.1 if args.obst_av else 1.0
initialization_std = 0.1
dim = (num_particles, ctl_generic.num_params)

# initial_particles = Normal(0, initialization_std).sample(dim).to(device)

initial_particles = empirical_ctl.get_parameters_as_vector_reduced().unsqueeze(dim=0)

# print("svg initial particles: ", initial_particles)
print(" *** SVGD initial particles: ", initial_particles[0, 0:50])

svgd_cont = SVGDCont(
    gibbs_posteior=gibbs_posteior,
    num_particles=num_particles, logger=logger,
    optimizer='Adam', lr=args.lr, lr_decay=None, #TODO: add decay
    initial_particles=initial_particles, kernel='RBF', bandwidth=None,
)
msg = '\n[INFO] SVGD: delta: %.2f' % args.delta + ' -- num particles: %2.f' % num_particles
msg += ' -- initialization std: %.4f' % initialization_std


# ****** TRAIN SVGD ******

logger.info('------------ Begin training ------------')
svgd_cont.fit(
    dataloader=train_dataloader,
    over_fit_margin=None, cont_fit_margin=None, max_iter_fit=None,
    early_stopping=early_stopping, log_period=args.log_epoch, epochs=args.epochs,
    valid_data=train_data   # NOTE: validate model using the entire train data
)
logger.info('Training completed.')


# # ------ Save trained model ------
# particles = svgd_cont.particles.detach().clone()
# res_dict = {
#     'particles':particles,
#     'num_rollouts':num_rollouts,
#     'Q':Q, 'alpha_u':alpha_u,
#     'alpha_ca':alpha_ca, 'alpha_obst':alpha_obst,
#     'n_xi':n_xi, 'l':l, 'initialization_std':initialization_std
# }
# # save file name
# if fname is not None:
#     filename_save = fname+'_'+str(num_particles)+'particles.pt'
# else:
#     filename_save = exp_name+'_SVGD_'+str(num_particles)+'particles_T'+str(t_end)+'_S'+str(num_rollouts)
#     filename_save += '_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pt'
# file_path = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', 'trained_models')
# path_exist = os.path.exists(file_path)
# if not path_exist:
#     os.makedirs(file_path)
# filename_save = os.path.join(file_path, filename_save)
# torch.save(res_dict, filename_save)
# logger.info('model saved.')

# eval on train data
bounded_train_loss = svgd_cont.eval_rollouts(train_data)
original_train_loss = svgd_cont.eval_rollouts(train_data, loss_fn=original_loss_fn)
logger.info('Final results on the entire train data: Bounded train loss = {:.4f}, original train loss = {:.4f}'.format(
    bounded_train_loss, original_train_loss
))

# ------------ 5. Test Dataset ------------

bounded_test_loss = svgd_cont.eval_rollouts(test_data)
original_test_loss = svgd_cont.eval_rollouts(test_data, loss_fn=original_loss_fn)
msg = 'True bounded test loss = {:.4f}, '.format(bounded_test_loss)
msg += 'true original test loss = {:.4f} '.format(original_test_loss)
msg += '(approximated using {:3.0f} test rollouts).'.format(test_data.shape[0])
logger.info(msg)

with torch.no_grad():
    x_log_test, u_log_test, dxref_test = sys.rollout(
        controller=ctl_generic, data=test_data
    )
    test_loss = original_loss_fn.forward(x_log_test, u_log_test)[0][0].item()   ## removed dxref_test
    print(f"\n TEST loss: {test_loss:.2f}")  ##

    x_log_train, u_log_train, dxref_train = sys.rollout(
        controller=ctl_generic, data=train_data
    )
    train_loss = original_loss_fn.forward(x_log_train, u_log_train)[0][0].item()    ## removed dxref_train
    print(f"\n TRAIN loss: {train_loss:.2f}")  ##

    x_log_test_plot, u_log_test_plot, dxref_test_plot = sys.rollout(
        controller=ctl_generic, data=test_data_plot
    )

plot_mode = "two"       # one if only plot train, two for testing and training datasets. "two_v2 is a failed attempt at plotting CI"

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
    plt.suptitle(f'System evolution with heat load', fontsize=17)

    plt.savefig(f'System evol, base controller, with demand .png')

    # plt.savefig("testing_no_load_no_PB.png")
    plt.show()


###### combined plots version 2 train and test
if plot_mode == "two":
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
    for i in range(x_log_train.shape[0]):
        axs[0, 0].plot(range(x_log_train.shape[1]), x_log_train[i] + T_ext.cpu(), color = train_color, alpha=alpha_value, marker=train_marker, label=f"Train {i}")
    axs[0, 0].set_title("X profile over the horizon")
    axs[0, 0].set_xlabel("Time (h)")
    axs[0, 0].set_ylabel("Temperature (°C)")
    axs[0, 0].grid()
    # axs[0, 0].axvline(x = 24,  color = 'r', linestyle='dashed')


    # Plot 2: DXref profile over the horizon
    for i in range( dxref_test_plot.shape[0]):  # Loop over test data rows
        axs[0, 1].plot(range(test_data_plot.shape[1]), dxref_test_plot[i], alpha=alpha_value, color = test_color, label=f"Test {i}")
    for i in range(dxref_train.shape[0]):  # Loop over train data rows
        axs[0, 1].plot(range(dxref_train.shape[1]), dxref_train[i], color=train_color, alpha=alpha_value,marker=train_marker, label=f"Train {i}")
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

    for i in range(u_log_train.shape[0]):  # Loop over train data rows
        axs[1, 1].plot(range(u_log_train.shape[1]), u_log_train[i], color=train_color, alpha=alpha_value, marker=train_marker, label=f"Train {i}")
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

    plt.suptitle(f'System evolution for lambda = {gibbs_lambda:.2f} (training loss: {original_train_loss:.2f}, test loss: {test_loss:.2f}), random seed = {args.random_seed}')

    plt.savefig("System evolution.png")
    # Show the figure
    plt.show()
    
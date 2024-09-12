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
from arg_parser import argument_parser, print_args
from plants import DHNDataset, DHNSystem
from assistive_functions import WrapLogger,heaviside
from loss_functions import DHNLoss


# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'LTI', 'saved_results')
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

print(args.random_seed)
#torch.manual_seed(args.random_seed)
#torch.manual_seed(8)


print(torch.seed())


# ------------ 1. Dataset ------------
disturbance = {
    'type':'normal noise',
}

dataset = DHNDataset(
    random_seed=args.random_seed, horizon=args.horizon,
    state_dim=args.state_dim, disturbance=disturbance,
    cp = 4186*10**(-6), mass = 200
)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=20)
train_data, test_data = train_data.to(device), test_data.to(device)



# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

sys = DHNSystem(
    mass=200,cop = 2
).to(device)

bys = [x / 100.0 for x in range(-10, 0, 1)]
val_loss_by = []
final_by = []
for by in bys:
    ctl = PerfBoostController(
        noiseless_forward=sys.noiseless_forward,
        input_init=sys.x_init, output_init=sys.u_init,
        dim_internal=args.dim_internal, dim_nl=args.l,initial_by=by,
        initialization_std=args.cont_init_std,
        output_amplification=20,
    ).to(device)


    # ------------ 4. Loss ------------
    #Size of the minimization 


    loss_fn = DHNLoss(
        R=args.alpha_u, u_min=dataset.umin, u_max=dataset.umax, x_min=dataset.xmin,x_max=dataset.xmax,
        alpha_xh = 5, alpha_uh=3,
        alpha_xl=7,
        alpha_ul = 7
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
            x_log, u_log, u2_log,u1_log,_ = sys.rollout(controller=ctl, data=train_data_batch)

            # loss of this rollout
            loss, loss_x, loss_u = loss_fn.forward(x_log, u_log, u2_log)

            # take a step
            loss_ul = torch.sum(loss_fn.f_lower_bound_u(u2_log),0)/x_log.shape[0]

            loss.backward()
            optimizer.step()





        # print info
        if epoch%args.log_epoch == 0:
            msg = 'Epoch: %i --- train loss: %.2f --- Loss xl : %.2f ---  loss u min: %.2f'% (epoch, loss, loss_x,loss_u)
            #msg +='--- Loss xh : %.2f ---  loss ul: %.2f---  loss uh: %.2f'% (loss_xh, loss_ul, loss_uh)
            if args.return_best:
                # rollout the current controller on the valid data
                with torch.no_grad():
                    x_log_valid, u_log_valid,u2_log_valid,u1_log_valid,dv = sys.rollout(
                        controller=ctl, data=valid_data
                    )

                if epoch == 0: 

                    x_log_test = x_log_valid.cpu()
                    u_log_test = u_log_valid.cpu()
                    u2_log_test = u2_log_valid.cpu()
                    delta = dv.cpu()



                    print("first")
                    plt.figure()
                    for i in range(valid_data.shape[0]): 
                        plt.plot(range(valid_data.shape[1]),u2_log_test[i])
                        plt.plot(range(valid_data.shape[1]),[2]*(valid_data.shape[1]), "--", c = "grey")
                        plt.plot(range(valid_data.shape[1]),[4]*(valid_data.shape[1]), "--",c = "grey" )
                        plt.title("U2 profile over the horizon")
                        plt.xlabel("Time (h)")
                        plt.ylabel("Temperature (°C)")
                    plt.savefig("saved_results/b_Y/u2_log_0_"+str(by)+".png")
                    plt.close()

                    plt.figure()
                    for i in range(valid_data.shape[0]): 
                        plt.plot(range(valid_data.shape[1]),delta[i])
                        plt.title("Delta profile over the horizon")
                        plt.xlabel("Time (h)")
                        plt.ylabel("Temperature (°C)")
                    plt.savefig("saved_results/b_Y/delta_log_0"+str(by)+".png")
                    plt.close()


                    plt.figure()
                    for i in range(valid_data.shape[0]):     
                        plt.plot(range(valid_data.shape[1]),u_log_test[i])
                        plt.plot(range(valid_data.shape[1]),[2]*(valid_data.shape[1]), "--", c = "grey")
                        plt.plot(range(valid_data.shape[1]),[4]*(valid_data.shape[1]), "--",c = "grey" )
                        plt.title("U profile over the horizon")
                        plt.xlabel("Time (h)")
                        plt.ylabel("Energy (MJ)")
                    plt.savefig("saved_results/b_Y/u_profile_0"+str(by)+".png")
                    plt.close()


                # loss of the valid data
                loss_valid, loss_x_v, loss_u_v = loss_fn.forward(x_log_valid, u_log_valid,u2_log_valid)

                loss_ul_v = loss_fn.alpha_ul*torch.sum(loss_fn.f_lower_bound_u(u2_log),0)/x_log.shape[0]
                loss_uh_v = loss_fn.alpha_uh*torch.sum(loss_fn.f_upper_bound_u(u2_log),0)/x_log.shape[0]

                msg += ' ---||--- validation loss: %.2f  --- Loss x low: %.2f ---  loss u min: %.2f ---  loss u low: %.2f---  loss u high: %.2f' % (loss_valid,loss_x_v,loss_u_v,loss_ul_v,loss_uh_v)
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

    val_loss_by.append(best_valid_loss)
    final_by.append(ctl.c_ren.b_y.cpu().detach().numpy()[0,0])

plt.figure()
plt.plot(bys,val_loss_by)
plt.title("Validation loss for different initial biases")
plt.xlabel("Initial by")
plt.ylabel("Validation Loss")
plt.savefig("saved_results/valid_loss.png")


plt.figure()
plt.plot(bys,final_by)
plt.title("Final bias for different initial biases")
plt.xlabel("Initial by")
plt.ylabel("Final by")
plt.savefig("saved_results/final_by.png")




with torch.no_grad():
    x_log_test, u_log_test,u2_log_test,u1_log_test,delta = sys.rollout(
        controller=ctl, data=test_data
    )



lower_bound_losses = loss_fn.f_lower_bound_x(x_batch=x_log_test,s = False).cpu()
lower_bound_u = loss_fn.f_lower_bound_u(u_batch=u2_log_test+2,s = False).cpu()

x_log_test = x_log_test.cpu()
u_log_test = u_log_test.cpu()
u2_log_test = u2_log_test.cpu()
delta = delta.cpu()


print("U1")
print(u1_log_test[1,:,:])


print("U2")
print(u2_log_test[1,:,:])

print("Delta")
print(delta[1,:,:])
plt.figure()
for i in range(test_data.shape[0]): 
    plt.plot(range(test_data.shape[1]),lower_bound_losses[i])
    plt.title("Loss X profile over the horizon")
    plt.xlabel("Time (h)")
    plt.ylabel("Loss X")
plt.savefig("saved_results/lower_bound_loss.png")



plt.figure()
for i in range(test_data.shape[0]): 
    plt.plot(range(test_data.shape[1]),lower_bound_u[i])
    plt.title("Loss  U profile over the horizon")
    plt.xlabel("Time (h)")
    plt.ylabel("Loss U")
plt.savefig("saved_results/lower_bound_u.png")



plt.figure()
for i in range(test_data.shape[0]): 
    plt.plot(range(test_data.shape[1]),x_log_test[i]+25)
    plt.plot(range(test_data.shape[1]),[40]*(test_data.shape[1]), "--", c = "grey")
    plt.plot(range(test_data.shape[1]),[80]*(test_data.shape[1]), "--",c = "grey" )
    plt.title("X profile over the horizon")
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature (°C)")
plt.savefig("saved_results/x_log.png")

plt.figure()
for i in range(test_data.shape[0]): 
    plt.plot(range(test_data.shape[1]),u2_log_test[i])
    plt.plot(range(test_data.shape[1]),[2]*(test_data.shape[1]), "--", c = "grey")
    plt.plot(range(test_data.shape[1]),[4]*(test_data.shape[1]), "--",c = "grey" )
    plt.title("U2 profile over the horizon")
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature (°C)")
plt.savefig("saved_results/u2_log.png")

plt.figure()
for i in range(test_data.shape[0]): 
    plt.plot(range(test_data.shape[1]),delta[i])
    plt.title("Delta profile over the horizon")
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature (°C)")
plt.savefig("saved_results/delta_log.png")


plt.figure()
for i in range(test_data.shape[0]):     
    plt.plot(range(test_data.shape[1]),u_log_test[i])
    plt.plot(range(test_data.shape[1]),[2]*(test_data.shape[1]), "--", c = "grey")
    plt.plot(range(test_data.shape[1]),[4]*(test_data.shape[1]), "--",c = "grey" )
    plt.title("U profile over the horizon")
    plt.xlabel("Time (h)")
    plt.ylabel("Energy (MJ)")
plt.savefig("saved_results/u_loss.png")


print(ctl.c_ren.b_y)
plt.show()
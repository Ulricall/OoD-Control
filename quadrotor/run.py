import quadsim
import controller
import trajectory
import numpy as np
import torch
import random
import argparse
import os
from bayes_opt import BayesianOptimization

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(C, Q, Name=""):
    print("Training " + Name)
    ace_error_list = np.empty(3)
    for i in range(3):
        setup_seed(i)
        if (Name == 'Neural-Fly'):
            C.wind_idx = i
        Wind_Velocity = np.random.normal(loc=0, scale=0.5*(i%3+1), size=(20,3))
        log = Q.run(trajectory = t, controller = C, wind_velocity_list = Wind_Velocity)
        log['p'] = log['X'][:, 0:3]
        squ_error = np.sum((log['p']-log['pd'])**2, 1)
        ace_error = np.mean(np.sqrt(squ_error))
        ace_error_list[i] = ace_error
    print("Training Error: ", np.mean(ace_error_list))
    return np.mean(ace_error_list)

def test(C, Q, Name, reset_control=True):
    print("Testing " + Name)
    C.state = 'test'
    ace_error_list = np.empty(2)
    for round in range(2):
        setup_seed(456+round*11)
        Wind_Velocity = np.random.uniform(low=-Wind_velo, high=Wind_velo, size=(20,3))
        # Wind_Velocity = np.random.normal(loc=0, scale=1, size=(20,3))
        log = Q.run(trajectory = t, controller = C, wind_velocity_list = Wind_Velocity, reset_control=reset_control, Name=Name)
        log['p'] = log['X'][:, 0:3]
        squ_error = np.sum((log['p']-log['pd'])**2, 1)
        if (args.logs):
            dir = 'logs/'+t.name
            if not os.path.exists(dir):
                os.makedirs(dir)
            np.save(dir+'/'+Name+'_'+str(round), log['p'])
        ace_error = np.mean(np.sqrt(squ_error))
        ace_error_list[round] = ace_error
    print("*******",Name,"*******")
    print("ACE Error: %.3f(%.3f)" % (np.mean(ace_error_list), np.std(ace_error_list, ddof=1)))
    return np.mean(ace_error_list)

def contrast_algo():
    # q_rl = quadsim.Quadrotor()
    # c_rl = controller.RLController(q_rl)
    # setup_seed(11)
    # c_rl.trace = t
    # c_rl.train()
    # test(c_rl, q_rl, "RL", False)

    # c_real = controller.RealMachine(eta_a_base=0.01, eta_A_base=0.05)
    # q_real = quadsim.Quadrotor()
    # train(c_real, q_real, "real_machine")
    # torch.save(c_real.phi, "model.pt")
    # torch.save(c_real.a, "a.npy")
    # test(c_real, q_real, "real_machine")

    # c_pid = controller.PIDController()
    c_deep = controller.MetaAdaptDeep(eta_a_base=0.01, eta_A_base=0.05)    
    # c_linear = controller.MetaAdaptLinear()
    # c_NF = controller.NeuralFly()
    q_pid = quadsim.Quadrotor()
    q_deep = quadsim.Quadrotor()
    # q_linear = quadsim.Quadrotor()
    # q_NF = quadsim.Quadrotor()
    train(c_deep, q_deep, "OMAC(deep)")
    # train(c_NF, q_NF, "Neural-Fly")
    # test(c_pid, q_pid, "PID")
    # test(c_linear, q_linear, "Linear")
    test(c_deep, q_deep, "OMAC(deep)", False)
    # test(c_NF, q_NF, "Neural-Fly", False)

def objfunction(x):
    if x==None: return None
    c_ood = controller.MetaAdaptOoD(eta_a_base=0.01, eta_A_base=0.05, noise_a=x, noise_x=x)
    q_ood = quadsim.Quadrotor()

    train(c_ood, q_ood, "OoD-Control")
    loss = test(c_ood, q_ood, "OoD-Control", False)
    return -loss  # bayes_opt only has maximize

def PIDobjfunc(p, i, d):
    c_pid = controller.PIDController(given_pid=True, p=p, i=i, d=d)
    q_pid = quadsim.Quadrotor()
    loss = train(c_pid, q_pid, "PID")
    return -loss


parser = argparse.ArgumentParser()
if __name__ == '__main__':
    parser.add_argument('--logs', type=int, default=1)
    parser.add_argument('--trace', type=str, default='hover')
    parser.add_argument('--wind', type=str, default='gale')
    args = parser.parse_args()
    if (args.wind=='breeze'):
        Wind_velo = 4
    elif (args.wind=='strong_breeze'):
        Wind_velo = 8
    elif (args.wind=='gale'):
        Wind_velo = 12
    else:
        raise NotImplementedError

    if (args.trace=='hover'):
        t = trajectory.hover()
    elif (args.trace=='fig8'):
        t = trajectory.fig8()
    elif (args.trace=='spiral'):
        t = trajectory.spiral_up()
    elif (args.trace=='sin'):
        t = trajectory.sin_forward()
    else:
        raise NotImplementedError
    
    # contrast_algo()
    optimizer = BayesianOptimization(
                    f=PIDobjfunc, 
                    pbounds={"p":(3, 9), 'i':(0,2), 'd':(1,5)},
                    verbose=2,
                    random_state=1,
                )
    optimizer.maximize(
                    init_points=2, 
                    n_iter=10,
                )
    best_p = optimizer.max['params']
    best_result = optimizer.max['target']
    print(best_p)
    print(-best_result)
    
    c_pid = controller.PIDController(given_pid=True, p=best_p['p'], i=best_p['i'], d=best_p['d'])
    q_pid = quadsim.Quadrotor()
    test_error = test(c_pid, q_pid, 'PID')
    print(test_error)
    # objfunction(0.06)
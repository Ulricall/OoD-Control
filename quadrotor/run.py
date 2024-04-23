import os
import random
import argparse
import torch
import numpy as np
import json

import quadsim
import controller
import trajectory
from bayes_opt import BayesianOptimization

# global variables to record the results
ACE_DICT = {}
STD_DICT = {}

def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params

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
        Wind_Velocity = np.random.gamma(shape=1., scale=0.5*(i%3+1), size=(20,3))
        log = Q.run(trajectory = t, controller = C, wind_velocity_list = Wind_Velocity)
        log['p'] = log['X'][:, 0:3]
        squ_error = np.sum((log['p']-log['pd'])**2, 1)
        ace_error = np.mean(np.sqrt(squ_error))
        ace_error_list[i] = ace_error
    print("Training Error: ", np.mean(ace_error_list))
    return np.mean(ace_error_list)

def test(C, Q, Name, reset_control=True):
    global ACE_DICT, STD_DICT

    print("Testing " + Name)
    C.state = 'test'
    ace_error_list = np.empty(10)
    for round in range(10):
        setup_seed(456+round*11)
        Wind_Velocity = np.random.uniform(low=-Wind_velo, high=0., size=(20,3))
        log = Q.run(trajectory = t, controller = C, wind_velocity_list = Wind_Velocity, 
                    reset_control=reset_control, Name=Name)
        log['p'] = log['X'][:, 0:3]
        squ_error = np.sum((log['p']-log['pd'])**2, 1)
        if (args.logs):
            dir = 'logs/'+t.name
            if not os.path.exists(dir):
                os.makedirs(dir)
            np.save(dir+'/'+Name+'_'+str(round), log['p'])
        ace_error = np.mean(np.sqrt(squ_error))
        ace_error_list[round] = ace_error
    ace = np.mean(ace_error_list)
    std = np.std(ace_error_list, ddof=1)
    ACE_DICT[Name] = ace
    STD_DICT[Name] = std
    print("*******", Name, "*******")
    print("ACE Error: %.3f(%.3f)" % (ace, std))
    return np.mean(ace_error_list)

def contrast_algo(given_pid=False, p=0, i=0, d=0):
    c_pid = controller.PIDController(given_pid=given_pid, p=p, i=i, d=d)
    c_linear = controller.MetaAdaptLinear(given_pid=given_pid, p=p, i=i, d=d)
    c_deep = controller.MetaAdaptDeep(given_pid=given_pid, p=p, i=i, d=d, 
                                      eta_a_base=0.01, eta_A_base=0.05)    
    c_NF = controller.NeuralFly(given_pid=given_pid, p=p, i=i, d=d)
    q_pid = quadsim.Quadrotor()
    q_deep = quadsim.Quadrotor()
    q_linear = quadsim.Quadrotor()
    q_NF = quadsim.Quadrotor()
    train(c_deep, q_deep, "OMAC(deep)")
    train(c_NF, q_NF, "Neural-Fly")
    test(c_pid, q_pid, "PID")
    test(c_linear, q_linear, "Linear")
    test(c_deep, q_deep, "OMAC(deep)", False)
    test(c_NF, q_NF, "Neural-Fly", False)

def objfunction(x=0.06, given_pid=False, p=0, i=0, d=0):
    if x==None: return None
    c_ood = controller.MetaAdaptOoD(given_pid=given_pid, p=p, i=i, d=d, 
                                    eta_a_base=0.01, eta_A_base=0.05, noise_a=x, noise_x=x)
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
    parser.add_argument('--use_bayes', type=bool, default=False)
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
    
    if (args.use_bayes):
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
    else:
        best_p = readparamfile('params/pid.json')
    
    contrast_algo(given_pid=True, p=best_p['p'], i=best_p['i'], d=best_p['d'])

    c_ood = controller.MetaAdaptOoD(given_pid=True, p=best_p['p'], i=best_p['i'], d=best_p['d'], 
                                    eta_a_base=0.01, eta_A_base=0.05, noise_a=0.06, noise_x=0.06)
    q_ood = quadsim.Quadrotor()

    train(c_ood, q_ood, "OoD-Control")
    test(c_ood, q_ood, "OoD-Control", False)
import torch
import controller
import simulation
import numpy as np
import random
import argparse
import os
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(Q, Name=''):
    losses = []
    for i in range(3):
        if (Name == 'Neural-Fly'):
            Q.controller.wind_idx = i
        setup_seed(i)
        Wind = np.random.gamma(shape=1., scale=1., size=(20,2)) # Training wind distributions
        log, logf = Q.run(Wind=Wind)
        losses.append(np.mean(np.abs(log[:,0])))
    return np.mean(np.array(losses))

def test(Q, Name):
    Q.test = True
    losses = []
    for i in range(10):
        setup_seed(100+i)
        Wind = np.random.uniform(low=-Wind_velo, high=0, size=(20,2)) # Test wind distributions
        # Wind = np.random.uniform(low=-Wind_velo, high=Wind_velo, size=(20,2)) 
        # Wind = np.random.normal(loc=0, scale=1, size=(20,2))
        log, logf = Q.run(Wind=Wind)
        losses.append(np.mean(np.abs(log[:,0])))
        if (args.logs==1):
            if not os.path.exists('./logs'):
                os.makedirs('./logs')
            if not os.path.exists('./F_logs'):
                os.makedirs('./F_logs')
            np.save('logs/'+Name+'_'+str(i)+'.npy', log)
            np.save('F_logs/'+Name+'_'+str(i)+'.npy', logf)
    losses = np.array(losses)
    print(Name, "ACE Error: %.3f(%.3f)" % (np.mean(losses), np.std(losses, ddof=1)))
    return np.mean(losses)

def contrast_algos(given_pid=False, p=0, i=0, d=0):
    c_pid = controller.PIDController(given_pid=given_pid, p=p, i=i, d=d)
    q_pid = simulation.Pendulum(c_pid)
    c_linear = controller.MetaAdaptLinear(given_pid=given_pid, p=p, i=i, d=d, eta_a=0.04)
    q_linear = simulation.Pendulum(c_linear)
    c_deep = controller.MetaAdaptDeep(given_pid=given_pid, p=p, i=i, d=d, eta_a=0.04, eta_A=0.02)
    q_deep = simulation.Pendulum(c_deep)
    c_neural = controller.NeuralFly(given_pid=given_pid, p=p, i=i, d=d, eta_a=0.04, eta_A=0.02)
    q_neural = simulation.Pendulum(c_neural)
    
    train(q_deep)
    train(q_linear)
    train(q_neural)
    test(q_pid, 'PID')
    test(q_linear, 'Linear')
    test(q_deep, 'OMAC(deep)')
    test(q_neural, 'NeuralFly')

def objfunc(noise_x):
    c_ood = controller.MetaAdaptOoD(eta_a=0.04, eta_A=0.02, noise_x=noise_x)
    q_ood = simulation.Pendulum(c_ood)
    loss = train(q_ood, "OoDControl")
    return -loss

def PIDobjfunc(p, i, d):
    c_pid = controller.PIDController(given_pid=True, p=p, i=i, d=d)
    q_pid = simulation.Pendulum(c_pid)
    loss = train(q_pid, "PID")
    return -loss

parser = argparse.ArgumentParser()
if __name__=='__main__':
    parser.add_argument('--logs', type=int, default=1)
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

    '''
    # Bayesian Optimization for PID
    optimizer = BayesianOptimization(
                    f=PIDobjfunc, 
                    pbounds={"p":(2, 6), 'i':(0, 1), 'd':(2, 6)},
                    verbose=2,
                    random_state=1,
                )
    optimizer.maximize(
                init_points=2,
                n_iter=7,
            )
    best_p = optimizer.max['params']
    best_result = optimizer.max['target']
    
    '''
    contrast_algos(given_pid=True, p=4.452, i=0.0, d=3.284)
    '''
    
    # Bayesian Optimization for the noise of OoD-Control
    optimizer_ood = BayesianOptimization(
                    f=objfunc,
                    pbounds={"noise_x":(0, 0.005)},
                    verbose=2,
                    random_state=1,
                )
    optimizer_ood.maximize(
                init_points=2,
                n_iter=7,
            )
    
    best_p_ood = optimizer_ood.max['params']
    best_result_ood = optimizer_ood.max['target']
    
    # OoD-Control
    c_ood = controller.MetaAdaptOoD(given_pid=True, p=4.452, i=0.0, d=3.284,
                                eta_a=0.04, eta_A=0.02, noise_x=7.415e-05)
    q_ood = simulation.Pendulum(c_ood)
    train_adversarial_attack(q_ood)
    test_adversarial_attack(q_ood, 'OoDControl')
    '''
    
    # Experiments for contrast algorithms
    # contrast_algos(given_pid=True, p=4.452, i=0.0, d=3.284)

    '''
    # Bayesian Optimization for the noise of OoD-Control
    optimizer_ood = BayesianOptimization(
                    f=objfunc,
                    pbounds={"noise_x":(0, 0.005)},
                    verbose=2,
                    random_state=1,
                )
    optimizer_ood.maximize(
                init_points=2,
                n_iter=7,
            )
    
    best_p_ood = optimizer_ood.max['params']
    best_result_ood = optimizer_ood.max['target']
    '''
    
    # OoD-Control test
    c_ood = controller.MetaAdaptOoD(given_pid=True, p=4.452, i=0.0, d=3.284, eta_a=0.04, eta_A=0.02, 
                                    noise_x=7.415e-05)
    q_ood = simulation.Pendulum(c_ood)
    train(q_ood)
    test(q_ood, 'OoDControl')
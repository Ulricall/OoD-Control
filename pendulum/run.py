import torch
import controller
import simulation
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(Q, Name=''):
    for i in range(3):
        if (Name == 'Neural-Fly'):
            Q.controller.wind_idx = i
        setup_seed(i)
        Wind = np.random.normal(loc=0, scale=1, size=(20,2))
        Q.run(Wind=Wind)

def test(Q, Name):
    Q.test = True
    losses = []
    for i in range(10):
        setup_seed(100+i)
        Wind = np.random.uniform(low=-4, high=4, size=(20,2))
        log, logf = Q.run(Wind=Wind)
        losses.append(np.mean(np.abs(log[:,0])))
        np.save('logs/'+Name+'_'+str(i)+'.npy', log)
        np.save('F_logs/'+Name+'_'+str(i)+'.npy', logf)
        #print(losses[-1])
    #print(losses)
    losses = np.array(losses)
    print("ACE Error: %.3f(%.3f)" % (np.mean(losses), np.std(losses, ddof=1)))

def objfunc(noise_x, noise_a):
    c_pid = controller.PIDController()
    q_pid = simulation.Pendulum(c_pid)
    c_linear = controller.MetaAdaptLinear(eta_a=0.04)
    q_linear = simulation.Pendulum(c_linear)
    c_deep = controller.MetaAdaptDeep(eta_a=0.04, eta_A=0.02)
    q_deep = simulation.Pendulum(c_deep)
    c_ood = controller.MetaAdaptOoD(eta_a=0.04, eta_A=0.02, noise_x=noise_x, noise_a=noise_a)
    q_ood = simulation.Pendulum(c_ood)
    c_neural = controller.NeuralFly(eta_a=0.04, eta_A=0.02)
    q_neural = simulation.Pendulum(c_neural)

    train(q_deep)
    train(q_ood)
    train(q_linear)
    train(q_neural)

    test(q_linear, 'Linear')
    test(q_pid, 'PID')
    test(q_deep, 'OMAC(deep)')
    test(q_ood, 'OoDControl')
    test(q_neural, 'NeuralFly')

if __name__=='__main__':
    for noise_x in [0.0003]:
        for noise_a in [0.0003]:
            objfunc(noise_x, noise_a)
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data_F(Name):
    logs = []
    for i in range(10):
        log = np.load('F_logs/'+Name+'_'+str(i)+'.npy', allow_pickle=True)
        logs.append(log)
    logs = np.array(logs)
    return np.sum(logs, axis=0) / 10

def load_data_p(Name):
    logs = []
    for i in range(10):
        log = np.load('logs/'+Name+'_'+str(i)+'.npy', allow_pickle=True)
        logs.append(log)
    logs = np.array(logs)
    return np.sum(logs, axis=0) / 10

def Plot_F(Name):
    log = load_data_F(Name)
    t = np.linspace(0, 30, log.shape[0])
    zero = np.zeros((log.shape[0]))
    plt.ylim((-22, 2))
    plt.plot(t, -log[:,0], c='r', label=r'$\hat{f}$')
    plt.plot(t, log[:,1], '--', c='b', label=r'$f$')
    plt.plot(t, zero, '--', c='k')
    plt.legend()
    if not os.path.exists('./pics'):
        os.makedirs('./pics')
    plt.savefig('pics/'+Name+'_torque.eps')
    plt.close()

def Plot_p(Name):
    log = load_data_p(Name)
    t = np.linspace(0, 30, log.shape[0])
    zero = np.zeros((log.shape[0]))
    plt.ylim((-1.2, 1.2))
    plt.plot(t, log[:,0], c='r', label=r'$\theta$')
    plt.plot(t, log[:,1], '--', c='b', label=r'$\dot{\theta}$')
    plt.plot(t, zero, '--', c='k')
    plt.legend()
    if not os.path.exists('./pics'):
        os.makedirs('./pics')
    plt.savefig('pics/'+Name+'_state.eps')
    plt.close()

if __name__ == '__main__':
    for Name in ['OoDControl', 'PID', 'OMAC(deep)', 'NeuralFly', 'Linear']:
        Plot_p(Name)
        Plot_F(Name)
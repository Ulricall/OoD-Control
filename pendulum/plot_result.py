import matplotlib.pyplot as plt
import numpy as np

def load_data(Name):
    logs = []
    for i in range(10):
        log = np.load('F_logs/'+Name+'_'+str(i)+'.npy', allow_pickle=True)
        logs.append(log)
    logs = np.array(logs)
    return np.sum(logs, axis=0) / 10

def Plot(Name):
    log = load_data(Name)
    t = np.linspace(0, 30, log.shape[0])
    zero = np.zeros((log.shape[0]))
    # print(log)
    plt.ylim((-20, 20))
    plt.plot(t, -log[:,0], c='r', label=r'$\hat{f}$')
    plt.plot(t, log[:,1], '--', c='b', label=r'$f$')
    #plt.plot(t, zero, '--', c='k')
    plt.legend()
    plt.savefig('pics/'+Name+'.eps')
    plt.show()
    plt.close()

if __name__ == '__main__':
    for Name in ['OoDControl', 'PID', 'OMAC(deep)', 'NeuralFly', 'Linear']:
        Plot(Name)
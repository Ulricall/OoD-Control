import matplotlib.pyplot as plt
import numpy as np
import trajectory

def load_data(Name):
    logs = []
    for i in range(10):
        log = np.load('logs/'+Name+'_'+str(i)+'.npy', allow_pickle=True)
        logs.append(log)
    logs = np.array(logs)
    return np.sum(logs, axis=0) / 10

dt = 0.01
def plot_3D_trace(Names):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x[m]")
    ax.set_ylabel("y[m]")
    ax.set_zlabel("z[m]")

    colors = ['r', 'g', 'b', 'y', 'm']

    t = trajectory.fig8()
    gt = np.zeros((2000,3))
    for i in range(2000):
        pd, vd, ad = t(i*dt)
        gt[i, :] = pd
    ax.plot(gt[:,0], gt[:,1], gt[:,2], '--')

    for i,name in enumerate(Names):
        log = load_data(name)
        ax.plot(log[:,0], log[:,1], log[:,2], c=colors[i], label=name)
    
    plt.legend(loc='upper right')
    plt.savefig("trace.eps", dpi=150)
    plt.show()

if __name__=='__main__':
    plot_3D_trace(['PID', "OoD-Control", 'OMAC(deep)', 'Linear', 'Neural-Fly'])

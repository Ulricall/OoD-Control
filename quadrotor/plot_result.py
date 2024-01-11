import matplotlib.pyplot as plt
import numpy as np
import trajectory
import matplotlib
import argparse
import os

t = trajectory.sin_forward()
def load_data(Name):
    logs = []
    for i in range(10):
        log = np.load('logs/'+t.name+'/'+Name+'_'+str(i)+'.npy', allow_pickle=True)
        logs.append(log)
    logs = np.array(logs)
    return np.sum(logs, axis=0) / 10

dt = 0.01

def get_ground_truth(t, len):
    seq_len = int(len/dt+1)
    gt = np.zeros((seq_len,3))
    for i in range(seq_len):
        pd, vd, ad = t(i*dt)
        gt[i, :] = pd
    return gt

def plot_3D_trace(Names):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("x[m]")
    ax.set_ylabel("y[m]")
    ax.set_zlabel("z[m]")

    colors = ['r', 'g', 'b', 'y', 'm']

    gt = get_ground_truth(t, 20)
    ax.plot(gt[:,0], gt[:,1], gt[:,2], '--')

    for i,name in enumerate(Names):
        log = load_data(name)
        ax.plot(log[:,0], log[:,1], log[:,2], c=colors[i], label=name)

    plt.legend(loc='upper right')
    if not os.path.exists('./traces'):
        os.makedirs('./traces')
    plt.savefig("traces/"+t.name+".png", dpi=150)
    plt.close()
    #plt.show()

def show_project(name):
    gt = get_ground_truth(t, 20)
    plt.scatter(gt[:,0], gt[:,1], s=1, c='k', label='ground truth')
    log = load_data(name)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.3)
    plt.scatter(log[:,0], log[:,1], c=np.sqrt((log[:,0]-gt[:,0])**2+(log[:,1]-gt[:,1])**2), cmap='rainbow', label=name, s=2, norm=norm)
    plt.colorbar()
    plt.xlim((-1.2, 1.2))
    plt.ylim((-2.2, 0.2))
    #plt.legend(loc='upper right')
    if not os.path.exists('./projections'):
        os.makedirs('./projections')
    plt.savefig("projections/project_"+t.name+'_'+name+".eps", dpi=150)
    plt.close()
    #plt.show()

parser = argparse.ArgumentParser()
if __name__=='__main__':
    parser.add_argument('--trace', type=str, default='hover')
    args = parser.parse_args()
    if (args.trace=='hover'):
        t = trajectory.hover()
    elif (args.trace=='fig8'):
        t = trajectory.fig8()
    elif (args.trace=='spiral'):
        t = trajectory.spiral_up()
    elif (args.trace=='sin'):
        t = trajectory.sin_forward()
    Models = ['PID', "OoD-Control", 'OMAC(deep)', 'Linear', 'Neural-Fly']
    plot_3D_trace(Models)
    for model in Models:
        show_project(model)

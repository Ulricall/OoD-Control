import matplotlib.pyplot as plt
import numpy as np
import trajectory

t = trajectory.spiral_up()
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
    ax = fig.gca(projection='3d')
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
    plt.savefig("trace.eps", dpi=150)
    plt.show()

# def plot_2D(Names):
#     plt.xlabel('t[s]')
#     plt.ylabel('')
#     for i,name in enumerate(Names):
#         log = load_data(name)
#         plt.plot(logt, log[:,0])

def show_project(name):
    gt = get_ground_truth(t, 20)
    plt.scatter(gt[:,0], gt[:,1], s=1, c='k', label='ground truth')
    log = load_data(name)
    plt.scatter(log[:,0], log[:,1], c=np.sqrt((log[:,0]-gt[:,0])**2+(log[:,1]-gt[:,1])**2), cmap='rainbow', label=name, s=2)
    plt.legend(loc='upper right')
    plt.savefig("projections/project_"+t.name+'_'+name+".eps", dpi=150)

    #plt.show()

if __name__=='__main__':
    Models = ['PID', "OoD-Control", 'OMAC(deep)', 'Linear', 'Neural-Fly']
    for model in Models:
        show_project(model)

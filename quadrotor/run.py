import quadsim
import controller
import trajectory
import numpy as np
import torch
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(C, Q):
    Wind_Velocity = np.random.normal(loc=0, scale=1, size=(50,3))
    Q.run(trajectory = t, controller = C, wind_velocity_list = Wind_Velocity)

def test(C, Q, Name, reset_control=True):
    C.state = 'test'
    Wind_Velocity = np.random.uniform(low=-8, high=8, size=(50,3))
    log = Q.run(trajectory = t, controller = C, wind_velocity_list = Wind_Velocity, reset_control=reset_control)
    log['p'] = log['X'][:, 0:3]
    #print(log['p'])
    squ_error = np.sum((log['p']-log['pd'])**2, 1)
    rmse = np.sqrt(np.mean(squ_error))
    ace_error = np.mean(np.sqrt(squ_error))
    print("*******",Name,"*******")
    print("ACE Error: %.5f (%.5f)" % (ace_error,rmse))

def objfunction(x1, x2):
    noise_x = x1
    noise_a = x2

    c_pid = controller.PIDController()
    c_deep = controller.MetaAdaptDeep(eta_a_base=0.005, eta_A_base=0.05)
    c_ood = controller.MetaAdaptOoD(eta_a_base=0.005, eta_A_base=0.05, noise_a=noise_a, noise_x=noise_x)
    c_linear = controller.MetaAdaptLinear()
    q_pid = quadsim.Quadrotor()
    q_deep = quadsim.Quadrotor()
    q_ood = quadsim.Quadrotor()
    q_linear = quadsim.Quadrotor()

    train(c_deep, q_deep)
    train(c_ood, q_ood)
    
    test(c_pid, q_pid, "PID")
    test(c_linear, q_linear, "Linear")
    test(c_deep, q_deep, "OMAC(deep)", False)
    test(c_ood, q_ood, "OoD-Control", False)

if __name__ == '__main__':
    t = trajectory.hover([0,0,0])
    for noise_a in [0.01]:
        for noise_x in [0.01]:
            loss = objfunction(noise_x, noise_a)
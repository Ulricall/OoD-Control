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

def train(C, Q, Name=""):
    print("Training " + Name)
    for i in range(3):
        setup_seed(i)
        if (Name == 'Neural-Fly'):
            C.wind_idx = i
        Wind_Velocity = np.random.normal(loc=0, scale=0.5*(i+1), size=(20,3))
        #Wind_Velocity = np.random.uniform(low=-3*(i+1), high=3*(i+1), size=(20,3))
        Q.run(trajectory = t, controller = C, wind_velocity_list = Wind_Velocity)

def test(C, Q, Name, reset_control=True):
    print("Testing " + Name)
    C.state = 'test'
    ace_error_list = np.empty(10)
    for round in range(10):
        setup_seed(234+round*11)
        Wind_Velocity = np.random.uniform(low=-12, high=12, size=(20,3))
        log = Q.run(trajectory = t, controller = C, wind_velocity_list = Wind_Velocity, reset_control=reset_control, Name=Name)
        log['p'] = log['X'][:, 0:3]
        squ_error = np.sum((log['p']-log['pd'])**2, 1)
        np.save("logs/"+t.name+'/'+Name+"_"+str(round), log['p'])
        #rmse = np.sqrt(np.mean(squ_error))
        ace_error = np.mean(np.sqrt(squ_error))
        ace_error_list[round] = ace_error
    print("*******",Name,"*******")
    print("ACE Error: %.3f(%.3f)" % (np.mean(ace_error_list), np.std(ace_error_list, ddof=1)))

def objfunction(x1, x2):
    noise_x = x1
    noise_a = x2

    c_pid = controller.PIDController()
    c_deep = controller.MetaAdaptDeep(eta_a_base=0.01, eta_A_base=0.05)
    c_ood = controller.MetaAdaptOoD(eta_a_base=0.01, eta_A_base=0.05, noise_a=noise_a, noise_x=noise_x)
    c_linear = controller.MetaAdaptLinear()
    c_NF = controller.NeuralFly()
    q_pid = quadsim.Quadrotor()
    q_deep = quadsim.Quadrotor()
    q_ood = quadsim.Quadrotor()
    q_linear = quadsim.Quadrotor()
    q_NF = quadsim.Quadrotor()

    # q_rl = quadsim.Quadrotor()
    # c_rl = controller.RLController(q_rl)
    # c_rl.train()
    # test(c_rl, q_rl, "RL", False)

    train(c_deep, q_deep, "OMAC(deep)")
    train(c_ood, q_ood, "OoD-Control")
    train(c_NF, q_NF, "Neural-Fly")
    
    test(c_pid, q_pid, "PID")
    test(c_linear, q_linear, "Linear")
    test(c_deep, q_deep, "OMAC(deep)", False)
    test(c_ood, q_ood, "OoD-Control", False)
    test(c_NF, q_NF, "Neural-Fly", False)

if __name__ == '__main__':
    t = trajectory.hover()
    for noise_a in [0.01]:
        for noise_x in [0.01]:
            loss = objfunction(noise_x, noise_a)
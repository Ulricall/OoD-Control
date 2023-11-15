import torch
import numpy as np
import controller
import pkg_resources
import json

DEFAULT_PARAM_FILE = pkg_resources.resource_filename(__name__, 'params/pendulum.json')

standard = controller.PIDController()

def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params

class Pendulum:
    def __init__(self, controller=None, params_file = DEFAULT_PARAM_FILE):
        self.controller = controller
        self.params = readparamfile(params_file)
        self.m = 1.0
        self.l = 1.0
        self.g = 9.81
        self.dt = self.params['dt']
        self.test = False       # 0 for train and 1 for test

    def get_wind_force(self):
        wind_x = self.Wind[self.total_wind][0]
        wind_y = self.Wind[self.total_wind][1]
        v_x = self.l * self.state[1] * np.cos(self.state[0])
        v_y = self.l * self.state[1] * np.sin(self.state[0])
        R = np.array([wind_x - v_x, wind_y-v_y])
        F = self.params['Cd'] * np.linalg.norm(R) * R
        return self.l * np.sin(self.state[0]) * F[1] + self.l * np.cos(self.state[0]) * F[0] \
               - self.params['alpha']*self.state[1]
    
    def f(self, state):
        F_wind = self.get_wind_force()
        return np.array([state[1], (self.u + F_wind)/(self.m * self.l**2) + 
                         self.g / self.l * np.sin(state[0])])
    
    def step(self):
        if self.params['integration_method'] == 'rk4':
            k1 = self.f(self.state)
            k2 = self.f(self.state + k1 * self.dt/2)
            k3 = self.f(self.state + k2 * self.dt/2)
            k4 = self.f(self.state + k3 * self.dt)
            F_gt = self.get_wind_force()
            self.controller.inner_adapt(self.state, F_gt)
            self.state += (k1 + k2*2 + k3*2 + k4*2) * self.dt / 6
        else:
            Xdot = self.f(self.state)
            F_gt = self.get_wind_force()
            if self.test == True and self.params['adversarial_attack'] == True:
                self.controller.inner_adapt_adversarial_attack(self.state, F_gt, eps=0.5)
                self.state += Xdot * self.dt
            else:
                self.controller.inner_adapt(self.state, F_gt)
                self.state += Xdot * self.dt
    
    def run(self, init_theta=0., init_dtheta=0., duration=30, Wind=None):
        self.last_wind_state = 0
        self.Wind = Wind
        self.total_wind = 0
        self.state = np.array([init_theta, init_dtheta], dtype=float)

        t = 0
        Log = []
        logu = []
        while (t < duration):
            self.u = self.controller(self.state)
            # self.u = self.controller(self.controller.get_state_adv())
            logu.append([self.u - standard(self.state), self.get_wind_force()])
            Log.append(self.state.tolist())
            self.step()
            t += self.dt
            if (t - self.last_wind_state > self.params['wind_update_period']):
                self.total_wind += 1
                self.last_wind_state = t
                if (self.test == False):
                    self.controller.meta_adapt()
                else:
                    pass
        
        return np.array(Log), np.array(logu)

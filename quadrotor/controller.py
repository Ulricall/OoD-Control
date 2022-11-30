import torch
import numpy as np
import random
import pkg_resources
import rowan
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm

torch.set_default_tensor_type('torch.DoubleTensor')

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

DEFAULT_CONTROL_PARAM_FILE = pkg_resources.resource_filename(__name__, 'params/controller.json')
DEFAULT_PX4_PARAM_FILE = pkg_resources.resource_filename(__name__, 'params/px4.json')
DEFAULT_QUAD_PARAMETER_FILE = pkg_resources.resource_filename(__name__, 'params/quadrotor.json')

def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params

class Controller():
    def __init__(self, quadparamfile=DEFAULT_QUAD_PARAMETER_FILE, px4paramfile=DEFAULT_PX4_PARAM_FILE):
        self.params = readparamfile(quadparamfile)

        self.px4_params = readparamfile(px4paramfile) 
        self.px4_params['angrate_max'] = np.array((self.px4_params['MC_ROLLRATE_MAX'],
                                                  self.px4_params['MC_PITCHRATE_MAX'],
                                                  self.px4_params['MC_YAWRATE_MAX']))
        self.px4_params['angrate_gain_P'] = np.diag((self.px4_params['MC_ROLLRATE_P'],
                                                  self.px4_params['MC_PITCHRATE_P'],
                                                  self.px4_params['MC_YAWRATE_P']))
        self.px4_params['angrate_gain_I'] = np.diag((self.px4_params['MC_ROLLRATE_I'],
                                                  self.px4_params['MC_PITCHRATE_I'],
                                                  self.px4_params['MC_YAWRATE_I']))
        self.px4_params['angrate_gain_D'] = np.diag((self.px4_params['MC_ROLLRATE_D'],
                                                  self.px4_params['MC_PITCHRATE_D'],
                                                  self.px4_params['MC_YAWRATE_D']))
        self.px4_params['angrate_gain_K'] = np.diag((self.px4_params['MC_ROLLRATE_K'],
                                                  self.px4_params['MC_PITCHRATE_K'],
                                                  self.px4_params['MC_YAWRATE_K']))
        self.px4_params['angrate_int_lim'] = np.array((self.px4_params['MC_RR_INT_LIM'],
                                                   self.px4_params['MC_PR_INT_LIM'],
                                                   self.px4_params['MC_YR_INT_LIM']))
        self.px4_params['attitude_gain_P'] = np.diag((self.px4_params['MC_ROLL_P'],
                                                  self.px4_params['MC_PITCH_P'],
                                                  self.px4_params['MC_YAW_P']))
        self.px4_params['angacc_max'] = np.array(self.px4_params['angacc_max'])
        self.px4_params['J'] = np.array(self.px4_params['J'])
        self.B = None
        #self.reset_controller()
    
    def reset_controller(self):
        self.w_error_int = np.zeros(3)
        self.w_filtered = np.zeros(3)
        self.w_filtered_last = np.zeros(3)

    def limit(self, array, upper_limit, lower_limit=None):
        if lower_limit is None:
            lower_limit = - upper_limit
        array[array > upper_limit] = upper_limit[array > upper_limit]
        array[array < lower_limit] = lower_limit[array < lower_limit]
    
    def mixer(self, torque_sp, T_sp):
        #print(self.B)
        omega_squared = np.linalg.solve(self.B, np.concatenate(((T_sp,), torque_sp)))
        omega = np.sqrt(np.maximum(omega_squared, self.params['motor_min_speed']))
        omega = np.minimum(omega, self.params['motor_max_speed'])
        return omega 
    
    def attitude(self, q, q_sp):
        q_error = rowan.multiply(rowan.inverse(q), q_sp)
        omega_sp = 2 * self.px4_params['attitude_gain_P'] @ (np.sign(q_error[0]) * q_error[1:])
        self.limit(omega_sp, self.px4_params['angrate_max'])
        return omega_sp
    
    def angrate(self, w, w_sp, dt):
        w_error = w_sp - w
        #print("w_error", w_error)
        #print("w", w)
        self.w_error_int += dt * w_error
        self.limit(self.w_error_int, self.px4_params['angrate_int_lim'])

        const_w_filter = np.exp(- dt / self.px4_params['w_filter_time_const'])
        self.w_filtered *= const_w_filter
        self.w_filtered += (1 - const_w_filter) * w
        
        w_filtered_derivative = (self.w_filtered - self.w_filtered_last) / dt
        self.w_filtered_last[:] = self.w_filtered[:] # Python is a garbage language

        alpha_sp = self.px4_params['angrate_gain_K'] \
                    @ (self.px4_params['angrate_gain_P'] @ w_error 
                       + self.px4_params['angrate_gain_I'] @ self.w_error_int
                       - self.px4_params['angrate_gain_D'] @ w_filtered_derivative)
        self.limit(alpha_sp, self.px4_params['angacc_max'])
        return alpha_sp

class PIDController(Controller):
    def __init__(self, quadparamfile=DEFAULT_QUAD_PARAMETER_FILE, ctrlparamfile=DEFAULT_CONTROL_PARAM_FILE):
        super().__init__(quadparamfile=quadparamfile)
        self.params = readparamfile(filename=ctrlparamfile, params=self.params)
        #print(ctrlparamfile)

    def calculate_gains(self):
        self.params['K_i'] = np.array(self.params['K_i'])
        self.params['K_p'] = np.diag([self.params['Lam_xy']*self.params['K_xy'],
                       self.params['Lam_xy']*self.params['K_xy'],
                       self.params['Lam_z']*self.params['K_z']])
        self.params['K_d'] = np.diag([self.params['K_xy'], self.params['K_xy'], self.params['K_z']])
        self.B = np.array([self.params['C_T'] * np.ones(4), 
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., -1., 1., 1.]),
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., 1., 1., -1.]),
                           self.params['C_q'] * np.array([-1., 1., -1., 1.])])

    def reset_controller(self):
        super().reset_controller()
        self.calculate_gains()
        self.F_r_dot = None
        self.F_r_last = None
        self.t_last = None
        self.t_last_wind_update = -self.params['wind_update_period']
        self.p_error = np.zeros(3)
        self.v_error = np.zeros(3)
        self.int_error = np.zeros(3)
        self.dt = 0.
        self.dt_inv = 0.

    def get_q(self, F_r, yaw=0., max_angle=np.pi):
        q_world_to_yaw = rowan.from_euler(0., 0., yaw, 'xyz')
        rotation_axis = np.cross((0, 0, 1), F_r)
        if np.allclose(rotation_axis, (0., 0., 0.)):
            unit_rotation_axis = np.array((1., 0., 0.,))
        else:
            unit_rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_axis /= np.linalg.norm(F_r)
        rotation_angle = np.arcsin(np.linalg.norm(rotation_axis))
        if F_r[2] < 0:
            rotation_angle = np.pi - rotation_angle
        if rotation_angle > max_angle:
            rotation_angle = max_angle
        q_yaw_to_body = rowan.from_axis_angle(unit_rotation_axis, rotation_angle)

        q_r = rowan.multiply(q_world_to_yaw, q_yaw_to_body)
        return q_r
    
    def get_Fr(self, X, imu, pd, vd, ad, meta_adapt_trigger):
        p_error = X[0:3] - pd
        v_error = X[7:10] - vd
        self.int_error += self.dt * p_error
        a_r = - self.params['K_p'] @ p_error - self.params['K_d'] @ v_error - self.params['K_i'] @ self.int_error + ad
        F_r = (a_r * self.params['m']) + np.array([0., 0., self.params['m'] * self.params['g']])
        #print(F_r)

        if self.F_r_last is None:
            self.F_r_dot = np.zeros(3)
        else:
            lam = np.exp(-self.dt / self.params['force_filter_time_const'])
            self.F_r_dot *= lam
            self.F_r_dot += (1-lam) * (F_r - self.F_r_last) / self.dt
        self.F_r_last = F_r.copy()
        return F_r, self.F_r_dot

    def position(self, X, imu, pd, vd, ad, last_wind_update, t):
        if self.t_last is None:
            self.t_last = t
        else:
            self.dt = t - self.t_last
        if (self.t_last_wind_update < last_wind_update):
            self.t_last_wind_update = last_wind_update
            meta_adapt_trigger = True
        else:
            meta_adapt_trigger = False
        
        yaw = 0.
        self.t_last = t
        F_r, F_r_dot = self.get_Fr(X, imu=imu, pd=pd, vd=vd, ad=ad, meta_adapt_trigger=meta_adapt_trigger)
        T_r_prime = np.linalg.norm(F_r + self.params['thrust_delay'] * F_r_dot)
        q_r_prime = self.get_q(F_r + self.params['attitude_delay'] * F_r_dot, yaw)
        F_r_prime = rowan.to_matrix(q_r_prime) @ np.array((0, 0, T_r_prime))

        T_r_prime = np.linalg.norm(F_r_prime)
        q_r_prime = self.get_q(F_r_prime, yaw)
        return T_r_prime, q_r_prime

class MetaAdapt(PIDController):
    def __init__(self):
        super().__init__()
        self.motor_speed = np.zeros(4)
    
    def get_residual(self, X, imu):
        q = X[3:7]
        R = rowan.to_matrix(q)

        H = self.params['m'] * np.eye(3)
        G = np.array((0., 0., self.params['g'] * self.params['m']))
        T = self.params['C_T'] * sum(self.motor_speed ** 2)
        u = T * R @ np.array((0., 0., 1.))
        y = (H @ imu[0:3] + G - u)

        return y
    
    def get_Fr(self, X, imu, pd, vd, ad, meta_adapt_trigger):
        y = self.get_residual(X, imu)
        fhat_F = self.get_f_hat(X)
        #("residual", y)
        #print("fhat", fhat_F)
        self.inner_adapt(X, fhat_F, y)
        self.update_batch(X, fhat_F, y)
        if (meta_adapt_trigger and self.state=='train'):
            self.meta_adapt()
        
        Fr,Fr_dot = super().get_Fr(X, imu, pd, vd, ad, meta_adapt_trigger)
        f_hat = self.get_f_hat(X)
        return Fr-f_hat, Fr_dot
    
    def mixer(self, torque_sp, T_sp):
        self.motor_speed = super().mixer(torque_sp, T_sp)
        return self.motor_speed
    
    def get_f_hat(self,X):
        raise NotImplementedError
    def inner_adapt(self, X, fhat, y):
        raise NotImplementedError
    def update_batch(self, X, fhat, y):
        raise NotImplementedError
    def meta_adapt(self, ):
        raise NotImplementedError

class MetaAdaptDeep(MetaAdapt):
    class Phi(nn.Module):
        def __init__(self, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(13, layer_sizes[0]))
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    def __init__(self, dim_a=100, layer_size=(25,30), eta_a_base=0.001, eta_A_base=0.001):
        super().__init__()
        self.dim_a = dim_a - dim_a%3
        self.layer_sizes = layer_size
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base
        self.loss = nn.MSELoss()
        self.state = 'train'
    
    def reset_controller(self):
        super().reset_controller()
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(dim_kernel=self.dim_a//3, layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.inner_adapt_count = 0
        self.batch = []
    
    def get_phi(self, X):
        with torch.no_grad():
            return np.kron(np.eye(3), self.phi(torch.from_numpy(X)).numpy())
    
    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return phi @ self.a

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = self.eta_a_base / np.sqrt(self.inner_adapt_count)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)

    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))
    
    def meta_adapt(self):
        self.inner_adapt_count = 0
        self.optimizer.zero_grad()
        loss = 0
        for X, y, a in self.batch:
            phi = torch.kron(torch.eye(3), self.phi(torch.from_numpy(X)))
            loss += self.loss(torch.matmul(phi, torch.from_numpy(a)), torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        self.batch = []

class MetaAdaptOoD(MetaAdaptDeep):
    def __init__(self, dim_a=100, layer_size=(25,30), eta_a_base=0.001, eta_A_base=0.001, noise_x=0.01, noise_a=0.01):
        super().__init__(dim_a=dim_a, layer_size=layer_size, eta_a_base=eta_a_base, eta_A_base=eta_A_base)
        self.noise_x = noise_x
        self.noise_a = noise_a
    
    def inner_adapt(self, X, fhat, y):
        self.a -= self.eta_a_base * 2 * (fhat - y).transpose() @ self.get_phi(X)

    def meta_adapt(self):
        self.optimizer.zero_grad()

        loss = 0
        for X, y, a in self.batch:
            X = X + self.noise_x*np.random.normal(0,1,X.shape)
            a = a + self.noise_a*np.random.normal(0,1,a.shape)
            phi = torch.kron(torch.eye(3), self.phi(torch.from_numpy(X)))
            loss += self.loss(torch.matmul(phi, torch.from_numpy(a)), torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        self.batch = []
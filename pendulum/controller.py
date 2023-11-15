import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm
import copy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Controller():
    def __init__(self):
        pass

    def __call__(self, state) :
        raise NotImplementedError

class RL():
    class Phi(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2,10)
            self.fc2 = nn.Linear(10,20)
            self.fc3 = nn.Linear(20,1)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    def __init__(self):
        self.m = self.l = 1
        self.g = 9.81
        self.phi = self.Phi()
        self.reward_delay = 0.95
    
    def get_wind_force(self, X, v_w):
        print(X)
        v_x = self.l * X[1] * np.cos(X[0])
        v_y = self.l * X[1] * np.sin(X[0])
        R = np.array([v_w[0] - v_x, v_w[1]-v_y])
        F = 0.3 * np.linalg.norm(R) * R
        return self.l * np.sin(X[0]) * F[1] + self.l * np.cos(X[0]) * F[0] - 0.5*X[1]
    
    def f(self, X, u, v):
        F_wind = self.get_wind_force(X, v)
        return np.array([X[1], (u+F_wind)/(self.m * self.l**2) + self.g/self.l * np.sin(X[0])])
    
    def calculate_reward(self):
        dt = 0.1
        reward = 0
        for T in range(5):
            gamma = 1
            t = 0
            vwind = np.random.normal(0, 1, (20,2))
            X = np.zeros(2)
            for i in range(50):
                u = self.phi(torch.from_numpy(X).float()).item()
                X += self.f(X, u, vwind[int(t),:]) * dt
                reward += gamma * (-X[0] - X[1])
                gamma *= self.reward_delay
                t += dt
        return reward / 5
    
    def train(self):
        reward = self.calculate_reward()
        for i in tqdm(range(3000)):
            self.phi_copy = copy.deepcopy(self.phi)
            gassian_kernel = torch.distributions.Normal(0, 1)
            with torch.no_grad():
                for param in self.phi.parameters():
                    param.mul_(torch.exp(gassian_kernel.sample(param.size())))
            new_reward = self.calculate_reward()
            if (new_reward < reward):
                self.phi = self.phi_copy
    
    def __call__(self, state):
        return self.phi(torch.from_numpy(state).float()).item()
    
    def inner_adapt(self, _, __):
        pass
    def meta_adapt(self):
        pass

class PIDController(Controller):
    def __init__(self, given_pid=False, p=0, d=0, i=0):
        self.integral = 0
        if (not given_pid):
            self.P = 3
            self.D = 3
            self.I = 0
        else:
            self.P = p
            self.D = d
            self.I = i
        self.m = self.l = 1
        self.g = 9.81
    
    def __call__(self, state):
        self.integral += state[0]
        return - self.m * self.l * self.g * np.sin(state[0]) - self.P*state[0] - self.I*self.integral - self.D*state[1]
    
    def inner_adapt(self, _, __):
        pass
    def meta_adapt(self):
        pass

class MetaAdapt(PIDController):
    def __init__(self, given_pid=False, p=0, d=0, i=0):
        super().__init__()
    
    def __call__(self, state):
        self.integral += state[0]
        return - self.m * self.l * self.g * np.sin(state[0]) - self.P*state[0] - self.I*self.integral - self.D*state[1] - self.f_hat(state)

class MetaAdaptLinear(MetaAdapt):
    def __init__(self, given_pid=False, p=0, d=0, i=0, eta_a=0.01, eta_A=0.01):
        super().__init__()
        setup_seed(345)
        self.W = np.random.uniform(low=-1, high=1, size=(20,2))
        self.a = np.random.normal(loc=0, scale=1, size=(20))
        # self.a = np.zeros(shape=self.dim_a)
        self.b = np.random.uniform(low=-1, high=1, size=(20))
        self.sub_step = 0
        self.eta_a = eta_a
        self.eta_A = eta_A
        self.batch = []
    
    def f_hat(self, state):
        return np.dot(self.W @ state + self.b, (self.a / np.linalg.norm(self.a)))
    
    def inner_adapt(self, state, y):
        self.sub_step += 1
        eta_a = self.eta_a / np.sqrt(self.sub_step)
        with torch.no_grad():
            fhat = self.f_hat(state)
            self.a -= eta_a * 2 * (fhat-y) * (self.W @ state + self.b)
        self.batch.append((state, self.a, y))
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20
    
    def meta_adapt(self):
        for X, a, y in self.batch:
            with torch.no_grad():
                fhat = self.f_hat(X)
            #print(np.outer(a,X))
            self.W -= self.eta_A * 2 * (fhat-y) * np.outer(a,X)
        self.batch = []
        self.sub_step = 0

class MetaAdaptDeep(MetaAdapt):
    class Phi(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(2,25))
            self.fc2 = spectral_norm(nn.Linear(25,50))
            self.fc3 = spectral_norm(nn.Linear(50,20))
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    def __init__(self, given_pid=False, p=0, d=0, i=0, eta_a=0.01, eta_A=0.01):
        super().__init__()
        self.eta_a = eta_a
        setup_seed(666)
        self.phi = self.Phi()
        self.optimizer = optim.Adam(self.phi.parameters(), lr=eta_A)
        self.a = np.zeros(20, dtype=float)
        self.sub_step = 0
        self.batch = []
        self.loss = nn.MSELoss()

        self.state_adv = np.zeros(2, dtype=float)
    
    def get_state_adv(self):
        return self.state_adv

    def inner_adapt_adversarial_attack(self, state, y, eps=10):
        state = torch.from_numpy(state).float()
        state.requires_grad = True
        temp = torch.dot(self.phi(state), torch.from_numpy(self.a).float())
        self.phi.zero_grad()
        temp_loss = self.loss(temp, torch.tensor(y, dtype=torch.float32))
        temp_loss.backward()
        state_adv = (state + eps * state.grad.sign()).detach().numpy()

        with torch.no_grad():
            fhat = self.f_hat(state_adv)
            self.a -= self.eta_a * 2 * (fhat - y) * self.phi(torch.from_numpy(state_adv).float()).numpy()
        '''
        with torch.no_grad():
            fhat = self.f_hat(state_adv)
            self.a -= eta_a * 2 * (fhat - y) * self.phi(torch.from_numpy(state_adv).float()).numpy()
        '''
        # self.batch.append((state_adv, self.a, y))
    
    def inner_adapt(self, state, y):
        self.sub_step += 1
        eta_a = self.eta_a / np.sqrt(self.sub_step)
        with torch.no_grad():
            fhat = self.f_hat(state)
            self.a -= eta_a * 2 * (fhat - y) * self.phi(torch.from_numpy(state).float()).numpy()
        self.batch.append((state, self.a, y))

    def meta_adapt(self):
        self.optimizer.zero_grad()
        loss = 0
        for X, a, y in self.batch:
            temp = torch.dot(self.phi(torch.from_numpy(X).float()), torch.from_numpy(a).float())
            loss += self.loss(temp, torch.tensor(y, dtype=torch.float32))
        loss.backward()
        self.optimizer.step()
        self.batch = []
        self.sub_step = 0
    
    def f_hat(self, state):
        return np.dot(self.phi(torch.from_numpy(state).float()).detach().numpy(), self.a)

class MetaAdaptOoD(MetaAdaptDeep):
    def __init__(self, given_pid=False, p=0, d=0, i=0, eta_a=0.01, eta_A=0.01, noise_x=0.025):
        super().__init__()
        self.noise_x = noise_x

        self.state_adv = np.zeros(2, dtype=float)

    def get_state_adv(self):
        return self.state_adv
    
    def inner_adapt(self, state, y):
        with torch.no_grad():
            fhat = self.f_hat(state)
            self.a -= self.eta_a * 2 * (fhat - y) * self.phi(torch.from_numpy(state).float()).numpy()
        self.batch.append((state, self.a, y))

    def inner_adapt_adversarial_attack(self, state, y, eps=10):
        state = torch.from_numpy(state).float()
        state.requires_grad = True
        temp = torch.dot(self.phi(state), torch.from_numpy(self.a).float())
        self.phi.zero_grad()
        temp_loss = self.loss(temp, torch.tensor(y, dtype=torch.float32))
        temp_loss.backward()
        state_adv = (state + eps * state.grad.sign()).detach().numpy()

        with torch.no_grad():
            fhat = self.f_hat(state_adv)
            self.a -= self.eta_a * 2 * (fhat - y) * self.phi(torch.from_numpy(state_adv).float()).numpy()

    def meta_adapt(self):
        self.optimizer.zero_grad()
        loss = 0
        for X, a, y in self.batch:
            X += self.noise_x * np.random.normal(loc=0, scale=1, size=X.shape)
            # a += self.noise_a * np.random.normal(loc=0, scale=1, size=a.shape)
            temp = torch.dot(self.phi(torch.from_numpy(X).float()), torch.from_numpy(a).float())
            loss += self.loss(temp, torch.tensor(y, dtype=torch.float32))
        
        loss.backward()
        self.optimizer.step()
        self.batch = []

class NeuralFly(MetaAdaptDeep):
    class H(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(20,30))
            self.fc2 = spectral_norm(nn.Linear(30,3))
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    def __init__(self, given_pid=False, p=0, d=0, i=0, eta_a=0.01, eta_A=0.01):
        super().__init__(given_pid, p, d, i, eta_a, eta_A)
        self.wind_idx = 0
        self.h = self.H()
        self.h_optimizer = optim.Adam(params=self.h.parameters(), lr=0.1)
        self.h_loss = nn.CrossEntropyLoss()

    def meta_adapt(self):
        self.inner_adapt_count = 0
        self.optimizer.zero_grad()
        loss = 0
        target = torch.tensor([self.wind_idx], dtype=int)
        for X, a, y in self.batch:
            temp = torch.dot(self.phi(torch.from_numpy(X).float()), torch.from_numpy(a).float())
            loss += self.loss(temp, torch.tensor(y, dtype=torch.float32))
        loss.backward()
        self.optimizer.step()

        if (np.random.uniform(0,1) < 0.5):
            loss_h = 0
            self.h_optimizer.zero_grad()
            for X, y, a in self.batch:
                phi = self.phi(torch.from_numpy(X).float()).detach()
                h = self.h(phi)
                loss_h += self.h_loss(h.unsqueeze(0), target)
            loss_h.backward()
            self.h_optimizer.step()
        self.batch = []
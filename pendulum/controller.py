import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch.optim as optim
import random

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

class PIDController(Controller):
    def __init__(self):
        self.integral = 0
        self.P = 3
        self.D = 3
        self.I = 0
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
    def __init__(self):
        super().__init__()
    
    def __call__(self, state):
        self.integral += state[0]
        return - self.m * self.l * self.g * np.sin(state[0]) - self.P*state[0] - self.I*self.integral - self.D*state[1] - self.f_hat(state)

class MetaAdaptLinear(MetaAdapt):
    def __init__(self, eta_a=0.01, eta_A=0.01):
        super().__init__()
        self.W = np.random.uniform(-1, 1, size=(20,2))
        self.a = np.random.uniform(-1, 1, size=(20))
        self.b = np.random.uniform(-1, 1, size=(20))
        self.sub_step = 0
        self.eta_a = eta_a
        self.eta_A = eta_A
        self.batch = []
    
    def f_hat(self, state):
        return np.dot(self.W @ state + self.b, self.a)
    
    def inner_adapt(self, state, y):
        self.sub_step += 1
        eta_a = self.eta_a / np.sqrt(self.sub_step)
        with torch.no_grad():
            fhat = self.f_hat(state)
            self.a -= eta_a * 2 * (fhat-y) * (self.W @ state + self.b)
        self.batch.append((state, self.a, y))
    
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

    def __init__(self, eta_a, eta_A):
        super().__init__()
        self.eta_a = eta_a
        setup_seed(666)
        self.phi = self.Phi()
        #self.phi.float()
        self.optimizer = optim.Adam(self.phi.parameters(), lr=eta_A)
        self.a = np.zeros(20, dtype=float)
        self.sub_step = 0
        self.batch = []
        self.loss = nn.MSELoss()
    
    def inner_adapt(self, state, y):
        self.sub_step += 1
        eta_a = self.eta_a / np.sqrt(self.sub_step)
        with torch.no_grad():
            fhat = self.f_hat(state)
            self.a -= eta_a * 2 * (fhat-y) * self.phi(torch.from_numpy(state).float()).numpy()
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
    def __init__(self, eta_a, eta_A, noise_x, noise_a):
        super().__init__(eta_a, eta_A)
        self.noise_x = noise_x
        self.noise_a = noise_a
    
    def inner_adapt(self, state, y):
        with torch.no_grad():
            fhat = self.f_hat(state)
            # print(fhat, y)
            self.a -= self.eta_a * 2 * (fhat-y) * self.phi(torch.from_numpy(state).float()).numpy()
        self.batch.append((state, self.a, y))

    def meta_adapt(self):
        self.optimizer.zero_grad()
        loss = 0
        for X, a, y in self.batch:
            X += np.random.normal(loc=0, scale=self.noise_x, size=X.shape)
            a += np.random.normal(loc=0, scale=self.noise_a, size=a.shape)
            temp = torch.dot(self.phi(torch.from_numpy(X).float()), torch.from_numpy(a).float())
            loss += self.loss(temp, torch.tensor(y, dtype=torch.float32))
        #print(loss)
        loss.backward()
        self.optimizer.step()
        self.batch = []

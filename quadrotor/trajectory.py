import numpy as np

class hover():
    def __init__(self, pd=np.zeros(3)):
        self.pd = pd
    def __call__(self, t):
        vd = ad = np.zeros(3)
        return self.pd, vd, ad

class fig8():
    _name = 'figure-8'
    def __init__(self, T=10., dir1=(2., 2., 0.), dir2=(0., 0., 1.)):
        self.const = 1/T
        self.w = self.const*2*np.pi
        self.dir1 = np.array(dir1)
        self.dir2 = np.array(dir2)

    def __call__(self, t):
        pd =                np.sin(self.w*t) * self.dir1 +                 np.sin(2*self.w*t) * self.dir2
        vd =   self.w     * np.cos(self.w*t) * self.dir1 +  2*self.w     * np.cos(2*self.w*t) * self.dir2
        ad = -(self.w)**2 * np.sin(self.w*t) * self.dir1 - (2*self.w)**2 * np.sin(2*self.w*t) * self.dir2
        return pd, vd, ad

class sin_forward():
    def __init__(self, T=6., A=2, Vy=0.2, Vz=0.5):
        self.w = np.pi*2/T
        self.A = A
        self.Vy = Vy
        self.Vz = Vz
    
    def __call__(self,t):
        pd = np.array((self.A*np.sin(self.w*t), self.Vy*t, self.Vz*t))
        vd = np.array((self.w * self.A * np.cos(self.w*t), self.Vy, self.Vz))
        ad = np.array((-self.w**2 * self.A * np.sin(self.w*t), 0, 0))
        return pd, vd, ad

class spiral_up():
    def __init__(self, T=5., R=1., Vz=0.8, Vr=0.2):
        self.w = np.pi*2/T
        self.R = R
        self.Vr = Vr
        self.Vz = Vz
    
    def __call__(self, t):
        pd = self.R*np.array((np.sin(self.w*t), np.cos(self.w*t)-1, self.Vz*t/self.R))
        vd = self.R*np.array((self.w * np.cos(self.w*t), -self.w * np.sin(self.w*t), self.Vz/self.R))
        ad = self.R*np.array((-self.w**2 * np.sin(self.w*t), -self.w**2 * np.cos(self.w*t), 0))
        return pd, vd, ad
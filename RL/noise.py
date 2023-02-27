import numpy as np
import tensorflow as tf

class OrnsteinUhlenbeck:
    def __init__(self, shape, dt, theta, mu, sigma, clip):
        self.dt=dt
        self.theta=theta
        self.mu=mu
        self.sigma=sigma
        self.clip = clip

        self.noise = np.empty((shape[0],), dtype = np.float64)
    
    def sample_noise(self):
        for index, item in enumerate(self.noise):
            self.noise[index] = (item + self.theta*(self.mu-item)*self.dt + 
                            (self.sigma*np.sqrt(self.dt)*np.random.normal(size=1)))
        self.noise = np.clip(self.noise, -self.clip, self.clip, dtype = np.float64)
        return self.noise






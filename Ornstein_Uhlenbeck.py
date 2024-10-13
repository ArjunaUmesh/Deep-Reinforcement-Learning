import numpy as np

class OUActionNoise() :
    
    def __init__(self,
                 mu,
                 sigma = 0.15,
                 theta = 0.2,
                 dt = 1e-2,
                 x0 = None):
        
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) + np.random.normal(size = self.mu.shape)
            
        self.prev_x = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        
        
##TEST class
# mu = np.zeros((2,3))        
# OUAN = OUActionNoise(mu=mu)
# t = OUAN()
# print(t)
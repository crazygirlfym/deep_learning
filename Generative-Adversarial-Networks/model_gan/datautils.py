import numpy as np

"""
Generate datas by Gaussian distribution
"""
class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5


    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

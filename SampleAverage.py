# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:11:22 2018

@author: Aidin
"""

import numpy as np

class kBandit:
    def __init__(self, k = 10, initial = 0 , reward = 0 , epsilon):
        # number of arms
        self.k = k 
        # instantanous reward
        self.reward = reward 
        # initialization for estimation
        self.initial = initial 
        # define the exploitation probability
        self.eps = epsilon
        # define set of arms
        self.arms = np.arange(self.k)
        # initialize time
        self.time = 0
        
    def initialize(self):
        # generating means for every arm
        self.mu = np.random.randn(self.k) + self.reward
        
        # initializing the estimated values
        self.muEstimated = np.zeros(self.k) + self.initial
        
        # initialize number of times each arm is played
        self.playCount = np.zeros(self.k)
        
        # best action
        self.bestAction =  np.argmax(self.mu)
        
    def algorithm(self):
        # exploration
        if np.random.rand() < self.eps:
            return np.random.choice(self.arms)
        
        # exploitation
        return np.argmax(self.muEstimated)
    
    def play(self,arm):
        # generate the reward for the played arm
        observedReward = np.random.randn() + self.mu[arm]
                
        # one step forward in time
        self.time += 1 
        
        # update the estimation with sample average algorithm
        self.muEstimated[arm] += 1.0 / self.playCount[arm] * (observedReward - self.muEstimated[arm])
        
        return observedReward
        
if __name__ == '__main__':
    
    
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:11:22 2018

@author: Aidin
"""



import matplotlib.pyplot as plt
import numpy as np



class kBandit:
    def __init__(self, epsilon, k = 10, initial = 0. , reward = 0. ):
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
        self.bestArm =  np.argmax(self.mu)
        
    def act(self):
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
        self.playCount[arm] += 1
        # update the estimation with sample average algorithm
        self.muEstimated[arm] += 1.0 / self.playCount[arm] * (observedReward - self.muEstimated[arm])
        return observedReward
    

def simulation(iterations, timeSteps, Bandits):
    bestArmCount = np.zeros((len(Bandits),iterations, timeSteps)) #total number of best arm played
    rewards = np.zeros(bestArmCount.shape) #average reward 
    bandit = Bandits #bandist defined here
    for b, bandit in enumerate(Bandits):
        for i in range(iterations):
            bandit.initialize()
            if i % 200 == 0:
                print ("iteration:" , i)
            for t in range(timeSteps):
                if i % 200 == 0 and t % 200 == 0:
                    print("Timestep" , t)
                arm = bandit.act()
                rewards[b,i,t] = bandit.play(arm)
                if arm == bandit.bestArm:
                    bestArmCount [b,i,t] = 1
    return bestArmCount.mean(axis = 1), rewards.mean(axis = 1)
            
if __name__ == '__main__':
    iterations = 4000
    timeSteps = 2000
    epss = [0, 0.1, 0.01]
    bandits = [kBandit(epsilon=eps) for eps in epss]
    bestArmCount, rewards= simulation(iterations, timeSteps, bandits)
    
    # draw the results
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 1, 1)
    for eps, rewards in zip(epss, rewards):
        plt.plot(rewards, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    
    plt.savefig('epsilonComparison.png')
    
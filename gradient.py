# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:11:22 2018

@author: Aidin
"""



import matplotlib.pyplot as plt
import numpy as np



class kBandit:
    def __init__(self, alpha = 0., k = 10, initial = 0. , reward = 0. ):
        # number of arms
        self.k = k 
        # instantanous reward
        self.reward = reward 
        # initialization for estimation
        self.initial = initial 
        # define the gradient timestep
        self.alpha = alpha
        # define set of arms
        self.arms = np.arange(self.k)
        # initialize time
        self.time = 0
        # define average reward
        self.averageReward = 0
        
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
        exponentials = np.exp(self.muEstimated)
        self.armProbability = exponentials / np.sum(exponentials)
        return np.random.choice(self.arms, p = self.armProbability)
    
    def play(self,arm):
        # generate the reward for the played arm
        observedReward = np.random.randn() + self.mu[arm]      
        # one step forward in time
        self.averageReward =  self.time /(self.time + 1) * self.averageReward + observedReward / (self.time + 1)
        self.time += 1 
        self.playCount[arm] += 1
        self.muEstimated[:arm] -= self.alpha * (observedReward - self.averageReward) * self.armProbability[:arm]
        self.muEstimated[arm+1:] -= self.alpha * (observedReward - self.averageReward) * self.armProbability[arm+1:]
        self.muEstimated[arm] += self.alpha * (observedReward - self.averageReward) * (1 - self.armProbability[arm])
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
    iterations = 2000
    timeSteps = 1000
    alphas = [0.1, 0.4, 0.7]
    bandits = [kBandit(alpha=a) for a in alphas]
    bestArmCount, rewards= simulation(iterations, timeSteps, bandits)
    
    # draw the results
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 1, 1)
    for alpha, rewards in zip(alphas, rewards):
        plt.plot(rewards, label='step size = %.02f' % (alpha))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    
    plt.savefig('stepSizeComparisonGradient.png')
    
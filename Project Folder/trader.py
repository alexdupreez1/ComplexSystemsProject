import random
import numpy as np

class Trader():
    """A generic class to represent a trader in the market."""
    def __init__(self):
        self.price = None

class Fundamentalist(Trader):
    def decide_price(self, pt, pf, phi, epsilon):
        """ 
        A Fundamentalist decides price based on the discrepancy from a fundamental value.
        Input: pt = price of current timestep, pf = fundamental price, 
               phi = weight of discrepancy, epsilon = noise
        """
        self.price = pt + phi * (pf - pt)+ epsilon

class Chartist(Trader):
    def decide_price(self, pt, pm, kappa, M, epsilon):
        """ 
        A Chartist decides price based on the average price of M past prices.
        Input: pt = price of current timestep, pm = average price of last M-values, M = the length of restrospective sight
               k = sensitivity of forecasts to past M prices, epsilon = noise
        """
        self.price = pt + kappa * (pt - pm)/M + epsilon

class RandomTrader(Trader):
    def decide_price(self, pt):
        """ 
        A random trader decides price without previous or fundamental price.
        Input: pt = price of current timestep
        """
        self.price = random.uniform(0, pt)

def global_price_compute(fundamentalists, chartists, random_traders, omega):
    """
    Calculate the global price
    Input: fundamentalist = a list of all fundamentalists
           chartists =  a list of all fundamentalists
           random_traders = a list of all random traders
    """
    N = len(fundamentalists) + len(chartists) + len(random_traders)
    F = len(fundamentalists)
    C = len(chartists)
    R = len(random_traders)
    
    pf_sum = sum([trader.price for trader in fundamentalists])
    pc_sum = sum([trader.price for trader in chartists])
    pr_sum = sum([trader.price for trader in random_traders])

    pg = (F/N) * pf_sum + (C/N) * pc_sum + (R/N) * pr_sum + omega
    return pg

# Test #
# Each list is buit by 20 corresponding instances
fundamentalists = [Fundamentalist() for _ in range(20)]
chartists = [Chartist() for _ in range(20)]
random_traders = [RandomTrader() for _ in range(20)]
current_price = 100
I_average = 0.4 # for test

# parameter setting in paper
fundamental_price = 5000 
phi = 2 
sigma = 200 
epsilon = random.uniform(-sigma, sigma)
kappa = 2    #normal distribution?
M = random.uniform(0, 90) # time step may be smaller than M ?
# I_average ?#
beta = 16 # exponent of global noise term
omega = epsilon * np.exp(beta * I_average)

for trader in fundamentalists:
    trader.decide_price(current_price, fundamental_price, phi, epsilon)

for trader in chartists:
    trader.decide_price(current_price, pm, kappa, M, epsilon)

for trader in random_traders:
    trader.decide_price(current_price)

global_price = global_price_compute(fundamentalists, chartists, random_traders, omega)
print(global_price)
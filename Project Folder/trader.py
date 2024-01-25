import random
import numpy as np

class Trader():
    """A generic class to represent a trader in the market."""
    def __init__(self,node_number):
        self.price = 0
        self.node_number = node_number
        self.type = None
        self.info_threshold = 1
        self.info = 0

class Fundamentalist(Trader):
    def __init__(self,node_number):
        super().__init__(node_number)
        self.type = 'fundamentalist'        
    def decide_price(self, pt, pf, phi, epsilon):
        """ 
        A Fundamentalist decides price based on the discrepancy from a fundamental value.
        Input: pt = price of current timestep, pf = fundamental price, 
               phi = weight of discrepancy, epsilon = noise
        """

        self.price = pt + phi * (pf - pt)+ epsilon

class Chartist(Trader):
    def __init__(self,node_number):
        super().__init__(node_number)
        self.type = 'chartist'   
    def decide_price(self, pt, pm, kappa, M, epsilon):
        """ 
        A Chartist decides price based on the average price of M past prices.
        Input: pt = price of current timestep, pm = average price of last M-values, M = the length of restrospective sight
               k = sensitivity of forecasts to past M prices, epsilon = noise
        """
        self.price = pt + kappa * (pt - pm)/M + epsilon

class RandomTrader(Trader):
    def __init__(self,node_number):
        super().__init__(node_number)
        self.type = 'random trader'
    def decide_price(self, pt):
        """ 
        A random trader decides price without previous or fundamental price.
        Input: pt = price of current timestep
        """
        self.price = random.uniform(0, pt)

def create_traders(num_traders, percent_fund, percent_chart):
    """Create a trader object list with each type of trader owning certain percentage."""

    num_fund = int(num_traders * percent_fund)
    num_chart = int(num_traders * percent_chart)
    num_rand = num_traders - num_fund - num_chart
    trader_types = ['fundamentalist'] * num_fund + ['chartist'] * num_chart + ['random trader'] * num_rand
    traders = [] # save all trader objets with different types
    for i in range(num_traders):
        random.shuffle(trader_types)
        trader_type = trader_types.pop()
        if trader_type == 'fundamentalist':
            traders.append(Fundamentalist(i))
        if trader_type == 'chartist':
            traders.append(Chartist(i))
        if trader_type == 'random trader':
            traders.append(RandomTrader(i)) 
    return traders

def global_price_calculate(traders, omega):
    """
    Calculate the global price
    Input: traders = a list of all traders
           omega = information-related noises
    """

    F = sum(1 for trader in traders if trader.type =='fundamentalist')
    C = sum(1 for trader in traders if trader.type =='chartist')
    R = sum(1 for trader in traders if trader.type =='random trader')
    N = F + C + R 

    
    pf_sum = sum(trader.price for trader in traders if trader.type =='fundamentalist')
    pc_sum = sum(trader.price for trader in traders if trader.type =='chartist')
    pr_sum = sum(trader.price for trader in traders if trader.type =='random trader')

    pg = (F/N) * pf_sum + (C/N) * pc_sum + (R/N) * pr_sum + omega
    return pg

# # parameter setting in paper
# fundamental_price = 5000 
# phi = 2 
# sigma = 200 
# epsilon = random.uniform(-sigma, sigma)
# kappa = 2    #normal distribution?
# M = random.uniform(0, 90) # time step may be smaller than M ?
# # I_average ?#
# beta = 16 # exponent of global noise term
# omega = epsilon * np.exp(beta * I_average)

import random
import numpy as np

class Trader():
    """A generic class to represent a trader in the market."""
    def __init__(self,node_number):
        self.price = 0
        self.node_number = node_number
        self.type = None
        self.info_threshold = 1
        self.info = np.random.uniform(0,1)
    

class Fundamentalist(Trader):
    """A class to represent a fundamentalist trader"""
    def __init__(self,node_number, phi):
        super().__init__(node_number)
        self.type = 'fundamentalist'    
        self.phi = phi #np.abs(np.random.normal(0,1)) #The paper uses 2 for phi!!
        #self.pf = np.random.uniform(1000,5000)
    def decide_price(self, pt, pf, epsilon):
        """ 
        A Fundamentalist decides price based on the discrepancy from a fundamental value.

        Input: pt = price of current timestep 
               pf = fundamental price 
               phi = weight of discrepancy 
               epsilon = noise
        """

        self.price = pt + self.phi * (pf - pt) + epsilon

class Chartist(Trader):
    """A class to represent a chartist trader"""
    def __init__(self,node_number):
        super().__init__(node_number)
        self.type = 'chartist'   
        self.m = np.random.randint(1,90) #Make this normally distributed later
        self.kappa = np.random.normal(0,1) 
    def decide_price(self, pt, pm, epsilon, m_current):
        """ 
        A Chartist decides price based on the average price of M past prices.
        Input: pt = price of current timestep, 
               pm = average price of last M-values, 
               M = the length of restrospective sight
               k = sensitivity of forecasts to past M prices, epsilon = noise
        """
        self.price = pt + self.kappa * ((pt - pm)/m_current ) + epsilon

class RandomTrader(Trader):
    """A class to represent a random trader"""
    def __init__(self,node_number):
        super().__init__(node_number)
        self.type = 'random trader'
    def decide_price(self, pt):
        """ 
        A random trader decides price without previous or fundamental price.
        Input: pt = price of current timestep
        """
        self.price = random.uniform(0.8*pt, 1.2*pt)

def create_traders(num_traders, percent_fund, percent_chart,phi):
    """Create a trader dictionary with a certain fraction of each type of trader"""

    num_fund = int(num_traders * percent_fund)
    num_chart = int(num_traders * percent_chart)
    num_rand = num_traders - num_fund - num_chart
    trader_types = ['fundamentalist'] * num_fund + ['chartist'] * num_chart + ['random trader'] * num_rand
    traders = [] 

    # generate the different fractions of traders
    for i in range(num_traders):
        random.shuffle(trader_types)
        trader_type = trader_types.pop()
        if trader_type == 'fundamentalist':
            traders.append(Fundamentalist(i,phi))
        if trader_type == 'chartist':
            traders.append(Chartist(i))
        if trader_type == 'random trader':
            traders.append(RandomTrader(i)) 
    return traders




def global_price_calculate(traders, sigma, beta):
    """ 
        Calculate the global price as a function of the traders's prices.
        Input: pt = price of current timestep, 
               pm = average price of last M-values, 
               M = the amount of steps considered
               k = sensitivity parameter
               epsilon = noise
        """
    epsilon = np.random.uniform(-1.5,1.5)

    exponent = beta * np.mean([traders[key].info for key in traders]) 

    omega = epsilon * np.exp(exponent)

    # average price of all traders.
    pg = np.mean([traders[key].price for key in traders]) + omega

    return pg

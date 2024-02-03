
import powerlaw
from itertools import chain
from networks import *
from plotting import *


def power_law_hypothesis_test(avalanches):

    """Performs a hypothesis test to compare the likelihood of an exponential distribution and a power law"""

    flattened_list = list(chain.from_iterable(avalanches))

    results = powerlaw.Fit(flattened_list)
    R, p = results.distribution_compare('power_law', 'exponential')

    return R, p


def multiple_runs_power_law_hypothesis_test(fundamental_percentage_list, network_params, networks ,max_info,global_prices,alpha,sigma, beta, pf, info_list,avalanches, num_days):

    """Performs a hypothesis test to compare the likelihood of an exponential distribution and a power law multiple times"""


    p_list = []
    R_list = []
    trader_configuration_list = []

    for funadamental_percentage in fundamental_percentage_list:
        global_prices = [5000]

        chartist_percentage = 1-funadamental_percentage
        network_params[3] = funadamental_percentage
        network_params[4] = chartist_percentage
        avalanches,avalanche_counter_current_time, avalanche_price_delta_list, global_prices, info_list, sum_avalanch_per_day = run_simulation(network_params, networks[0],max_info,global_prices,alpha,sigma, beta, pf, info_list,avalanches, num_days)        
        price_changes = calculate_price_changes_from_fundamental(global_prices,pf)
        results = powerlaw.Fit(price_changes)
        R, p = results.distribution_compare('power_law', 'exponential')
        p_list.append(p)
        R_list.append(R)
        trader_configuration_list.append((funadamental_percentage,chartist_percentage))

    return p_list, R_list





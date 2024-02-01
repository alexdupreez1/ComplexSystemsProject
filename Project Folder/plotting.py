import copy
import matplotlib.pyplot as plt
import numpy as np
from networks import *
import pandas as pd
from collections import Counter
from itertools import chain
from scipy.optimize import curve_fit
from scipy.stats import norm
import scipy.stats as stats
import powerlaw





#Price time series plotting 
#-----------------------------------------------------------------

def price_series(global_prices, filename):

    """Plots the price time series"""


    plt.figure(figsize=(24, 6))  
    plt.style.use('seaborn-darkgrid')  

    plt.plot(global_prices, linewidth=2, label='Global Prices', color='black')


    plt.axhline(y=5000, color='red', linestyle='--', linewidth=2, label='Fundamental Price')
    parts = filename.split('_')
    ratio_chartist, ratio_fundamentalist = float(parts[3]), float(parts[4])
    if round((ratio_chartist + ratio_fundamentalist)*100)!= 100:
        random = 1 - (ratio_chartist + ratio_fundamentalist)
        title = f"{round(ratio_chartist*100,0)}% Chartists & {round(float(ratio_fundamentalist)*100,0)}% Fundamentalists & {round(float(random)*100,0)}% Random Trader"                                       
    else:
        title = f"{round(ratio_chartist*100,0)}% Chartists & {round(float(ratio_fundamentalist)*100,0)}% Fundamentalists"
    
    plt.title(title, fontsize=18)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Price', fontsize=16)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    
    plt.savefig(filename +'price_series.svg', dpi=300)

def generate_price_plots(df):

    """Loops over the data frame with different ratios of traders and plots the price series"""

    for column in df.columns:
        global_prices = df[column].dropna().tolist()  
        filename = column  

        # Generate and save the price series plot
        price_series(global_prices, filename)


def create_data_frame(chartist_fundamentalist_ratio,networks,network_params, max_info, starting_prices, alpha, sigma, beta, pf, info_list, avalanches, num_days):

    # Initialize a dictionary to store the results
    simulation_results = {}

    for network_type in networks:
    # Loop over different chartist-fundamentalist ratios
        for ratio in chartist_fundamentalist_ratio:
        
        
            # Reset or reinitialize the variables for each simulation
            global_prices = copy.deepcopy(starting_prices)
            info_list = [[],[]]  # Reset information list
            avalanches = []         # Reset avalanches
            
            percent_chart, percent_fund = ratio
            network_params[-3] = percent_fund  # Update percent_fund in network_params
            network_params[-2] = percent_chart  # Update percent_chart in network_params

            # Run the simulation with the current parameters
            avalanches, avalanche_counter, price_deltas, global_prices, info_list, sum_avalanches = run_simulation(
                network_params,network_type, max_info, global_prices, alpha, sigma, beta, pf, info_list, avalanches, num_days)
            


            print('Running simulation for network: {} and ratio: {}'.format(network_type, ratio))
            
            # Construct the key for the current simulation
            key = f'{network_type}_ratio_{ratio[0]}_{ratio[1]}_global_prices'
            
            # Store the global prices for the current simulation
            simulation_results[key] = global_prices


    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in simulation_results.items()]))


    # Save to a CSV file
    df.to_csv('simulation_results.csv', index=False)

    return df



#Information SOC plotting 
#-----------------------------------------------------------------

#Average info and max info analysis

def plot_average_and_max_info(global_prices, info_list,avalanches):

    global_prices_sliced = global_prices[:500]
    info_list_sliced = info_list[0][:500]
    max_info_list_sliced = info_list[1][:500]

    avalanches_summed = [sum(day) for day in avalanches]

    avalanches_summed_sliced = avalanches_summed[:500]

    fig, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=(40, 20), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})

    ax2.plot(info_list_sliced, linewidth=2, color = 'black')
    ax2.set_ylabel('Average info', fontsize = 24)
    ax2.set_title('Average info vs avalanche volume', fontsize = 28)
    ax2.legend(fontsize = 12)
    ax2.tick_params(axis='y', labelsize=18)

    ax3.plot(max_info_list_sliced, linewidth=2, color = 'black')
    ax3.set_ylabel('Max info', fontsize = 24)
    ax3.set_title('Max info vs avalanche volume', fontsize = 28)
    ax3.legend(fontsize = 12)
    ax3.tick_params(axis='y', labelsize=18)

    bin_edges = np.arange(0, len(avalanches_summed_sliced) + 1, 1)

    ax4.hist(range(len(avalanches_summed_sliced)), bins=bin_edges, weights=avalanches_summed_sliced, color='gray')
    ax4.set_ylabel('Avalanche Volume', fontsize = 24)
    ax4.set_xlabel('Days', fontsize = 28)
    ax4.set_yscale('log')  
    ax4.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.savefig("BA_Out_10000T.svg", dpi = 300)

    plt.show()


# Information avalanche power law with small network

def plot_info_power_law_wattz(avalanches):

    flattened_list = list(chain.from_iterable(avalanches))
    value_counts = Counter(flattened_list)

    values = np.array(list(value_counts.keys()))
    counts = np.array(list(value_counts.values()))

    #indices = np.where(values<500) 

    #values = values[indices]
    #counts = counts[indices]

    non_zero_mask = counts > 0
    values_non_zero = values[non_zero_mask]
    counts_non_zero = counts[non_zero_mask]

    log_values = np.log(values_non_zero)
    log_counts = np.log(counts_non_zero)

    errorbars = 1 / np.sqrt(counts_non_zero)
    weights = np.sqrt(counts_non_zero)  

    def model_func(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(model_func, log_values, log_counts, sigma=errorbars, absolute_sigma=True)
    fitted_line = model_func(log_values, *popt)

    plt.figure(figsize=(10, 6))

    # Create a scatter plot on log-log scale with error bars
    plt.scatter(values_non_zero, counts_non_zero,alpha = 0.6,)
    plt.scatter(values_non_zero, counts_non_zero, color ='black',alpha = 0.5 ,facecolor = 'grey', label=f'Simulated data')
    plt.xlim(0.8,1000)
    plt.ylim(0.8,50000)
    plt.plot(values_non_zero, np.exp(fitted_line), color='black', label=f'Power-law fit (slope = {popt[0]:.2f})')

    # Set the scale to log for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Add labels and legend
    plt.title('Watts-Strogats Network', fontsize = 18)
    plt.xlabel('Avalanche sizes', fontsize = 16)
    plt.ylabel('Occurrences', fontsize = 16)
    plt.legend(fontsize = 12)
    #plt.savefig("WS_PL_10000T.svg", dpi = 300)
    plt.show()


# Information avalanche power law with random network
    
def plot_info_power_law_barabasi(avalanches):

    flattened_list = list(chain.from_iterable(avalanches))
    value_counts = Counter(flattened_list)
  
    values = np.array(list(value_counts.keys()))
    counts = np.array(list(value_counts.values()))

    non_zero_mask = counts > 0
    values_non_zero = values[non_zero_mask]
    counts_non_zero = counts[non_zero_mask]

    ind = np.where(np.max(values_non_zero))

    log_values = np.log(values_non_zero)
    log_counts = np.log(counts_non_zero)

    slope, intercept = np.polyfit(log_values, log_counts, 1)
    line = np.exp(intercept) * values_non_zero**(slope)

    int = log_values[ind] - slope*log_counts[ind]
    line1 = np.exp(int) * values_non_zero**(slope)

    errorbars = np.sqrt(counts_non_zero)

    plt.figure(figsize=(10, 6))
    plt.plot(values_non_zero, line, color='black', label=f'Power-law fit (slope = {slope:.2f}')
    plt.scatter(values_non_zero, counts_non_zero,  color ='black',alpha = 0.5 ,facecolor = 'grey', label=f'Simulated data')
    plt.xlim(0.8,1000)
    plt.ylim(0.8,50000)
    plt.xscale('log')
    plt.yscale('log')

    plt.title('Barabasi-Albert Network', fontsize = 18)
    plt.xlabel('Avalanche sizes', fontsize = 16)
    plt.ylabel('Occurrences', fontsize = 16)
    plt.legend(fontsize = 12)

    plt.show()


#Price SOC plotting 
#-----------------------------------------------------------------
    

#Price change from fundamental value power law 

def calculate_price_changes_from_fundamental(price_series, pf, step=1):
    price_changes = []
    for i in range(0, len(price_series) - step, step):
        change = np.abs(pf - price_series[i])
        price_changes.append(change)
    return price_changes

def plot_price_change_power_law(global_prices,pf):

    def power_law_formula(x, a, k):
        return a * np.power(x, -k)

    fig = plt.figure(figsize=(9, 6))
    price_changes = calculate_price_changes_from_fundamental(global_prices, pf)
    price_changes = np.array(price_changes)
    price_changes = price_changes[price_changes > 8]   
    
    bin_edges = np.arange(0, max(price_changes) + 0.8, 0.8)  # The edges of the bin intervals
    hist, bins = np.histogram(price_changes, bins=bin_edges, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Ensure we only fit to bins with non-zero counts
    non_zero_hist = hist[hist > 0]
    non_zero_bin_centers = bin_centers[hist > 0]
    
    weights = np.sqrt(non_zero_hist)
    # Fit the power law with cutoff
    popt, pcov = curve_fit(power_law_formula, non_zero_bin_centers, non_zero_hist, p0=(1.0, 1.0),sigma= weights,  maxfev=5000)
    a, k = popt

    # Prepare data for plotting fitted curve
    xspace = np.linspace(non_zero_bin_centers.min(), non_zero_bin_centers.max(), num=100)
    plt.scatter(non_zero_bin_centers, non_zero_hist, alpha=0.6, label='Data')
    plt.plot(xspace, power_law_formula(xspace, *popt), color='red', label=f'Powerlaw fit (k={k:.2f})')    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Price change')
    plt.ylabel('Frequency')
    plt.title('Price change from fundamental value power law')
    plt.legend()
    plt.show()
  
def powerlaw_package_fit(global_prices,pf):

    price_changes = calculate_price_changes_from_fundamental(global_prices, pf)
    results = powerlaw.Fit(price_changes)

    # The scaling parameter (alpha) of the power-law distribution corresponds to the exponent
    alpha = results.power_law.alpha
    sigma = results.power_law.sigma  # The standard error of the exponent
    sigma = results.power_law.xmin
    print("The exponent of the power law (alpha) is:", alpha)
    print("Standard error of the exponent:", sigma)
    print("The scaling parameter (xmin) is:", results.power_law.xmin)


def compute_normalised_returns(global_prices):

    returns = [global_prices[i] - global_prices[i-1] for i in range(len(global_prices)-1)]
    std = np.std(returns)
    mean = np.mean(returns)
    normalized_returns = [(returns[i]-mean)/std for i in range(len(returns))]

    return normalized_returns


def plot_fat_tail_returns(normalized_returns):


    # Normalize Returns    

    #PDF
    counts, edges = np.histogram(normalized_returns, bins=30, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    plt.scatter(bin_centers, counts, color = "b", label = "Normalised returns")

    # Standard Gaussian fitting curve
    mu = 0
    sigma = 1
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    Gaussian = norm.pdf(x, mu, sigma)
    plt.plot(x, Gaussian, label='Standard normal distribution', color = "black" , linestyle = '--')
    plt.yscale('log')

    plt.yscale('log')

    plt.ylim(bottom=np.min(np.array(normalized_returns)[np.array(normalized_returns) > 0]))  

    def fat_tail_model(x, A, B, q): #Defining a function to be fit
        return A * (1 - (1 - q) * B * x ** 2) ** (1 / (1 - q))


    ydata = np.array(normalized_returns)  

    ydata_sorted = np.sort(ydata)
    xdata = np.arange(1, len(ydata) + 1)

    xdata_normalized = xdata / len(ydata)

    initial_guess_A = max(ydata)
    initial_guess_B = 1 / (2 * np.var(ydata))
    initial_guess_q = 1.5 

    params, covariance = curve_fit(fat_tail_model, xdata_normalized, ydata_sorted, p0=[initial_guess_A, initial_guess_B, initial_guess_q])

    A, B, q = params

    xfit = np.linspace(min(xdata_normalized), max(xdata_normalized), 1000)
    yfit = fat_tail_model(xfit, A, B, q)


    plt.legend(loc='upper right', fontsize='small') 
    plt.xlabel("Normalised Returns")
    plt.ylabel("PDF")
    plt.show()
    print(q)


def plot_qq_plot(normalized_returns):

    stats.probplot(normalized_returns, dist="norm", plot=plt)
    plt.title('Quantile-Quantile Plot')
    plt.xlabel('Gaussian Quantiles')
    plt.ylabel('Returns Quantiles')
    plt.show()

def plot_p_values(fundamental_percentage_list, p_list, R_list):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    plt.subplots_adjust(hspace=0.5)  # Adjusts the vertical space between subplots

    ax1.plot(fundamental_percentage_list, R_list, marker = "o", color = "black", markersize=3)
    ax1.set_title("P and R values for different proportions of traders")
    ax1.set_xlabel("Percentages of fundamentalists")
    ax1.set_ylabel("R-values")
    ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    #ax1.set_yscale('log')
    #ax1.set_ylim(-100, 500)
    #ax1.set_xlim(0, 0.25)


    ax1.hlines(0, xmin=0, xmax=1, colors='black', linestyles='dashed')

    #ax1.set_yticks(range(int(min(R_list) * 1.1), int(max(R_list) * 1.1), 500))
    ax2.plot(fundamental_percentage_list, p_list, marker = "o", color = "black")
    ax2.set_xlabel("Percentages of fundamentalists")
    ax2.set_ylabel("P-values")
    #ax2.set_yscale('log')
    ax2.hlines(0.05, xmin=0, xmax=1, colors='r', linestyles='dashed', label = "Significance level")
    ax2.set_ylim(0, 0.06)
    ax2.set_yticks([0, 0.05, ax2.get_yticks()[-1]])
    ax2.legend(fontsize='small', loc= 'center right')






import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import random
from IPython.display import clear_output
from trader import *



def create_trader_network(num_traders, avg_degree, rewiring_probability,percent_fund,percent_chart, phi):

    """Creates a small world network of traders"""

    traders = create_traders(num_traders, percent_fund, percent_chart, phi)
    network = nx.watts_strogatz_graph(n=len(traders), k=avg_degree, p=rewiring_probability)
    
    for i, node in enumerate(network.nodes()):
        network.nodes[node]['trader'] = traders[i]

    trader_dictionary = {trader.node_number: trader for trader in traders}

    return network, trader_dictionary

def create_trader_network_barabasi(num_traders,percent_fund,percent_chart, phi):

    """Creates a small world network of traders"""

    traders = create_traders(num_traders, percent_fund, percent_chart, phi)
    network = nx.barabasi_albert_graph(num_traders, 3)
    
    for i, node in enumerate(network.nodes()):
        network.nodes[node]['trader'] = traders[i]

    trader_dictionary = {trader.node_number: trader for trader in traders}

    return network, trader_dictionary

def display_network(network, trader_dict):
    """Plots the network structure with additional information from trader_dict."""
    
    # Loop over the nodes and link them to a trader
    for node in network.nodes():
     
        if node in trader_dict:
            network.nodes[node]['label'] = trader_dict[node].type[0] + ": " + str(round(trader_dict[node].info,2))

            

    # Drawing the network
    pos = nx.spring_layout(network)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(network, pos, node_color='lightblue')

    # edges
    nx.draw_networkx_edges(network, pos, edge_color='red')

    nx.draw_networkx_labels(network, pos, labels=nx.get_node_attributes(network, 'label'))
    
    plt.show()

    
def get_neighbours(trader,network):

    """Gets the neighbours of a given node"""

    # get the neighbouring nodes
    neighbors = list(network.neighbors(trader.node_number)) 

    return neighbors

def add_global_info(trader_dictionary,max_info):

    """Adds global information to the network"""
    keys = list(trader_dictionary.keys())
    information_list = [trader_dictionary[key].info for key in keys]

    # distribute a randomly distributed amount of info to all traders
    for key in trader_dictionary:
        trader_dictionary[key].info += random.uniform(0 , max_info) 


def count_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2)
    return len(common_elements)


def neighbour_layer(current_layer,previous_layer, trader_dictionary, network, alpha, avalanche_size, expoloding_trader_price):
    
    """Handles the current layer of active traders"""
    
    next_layer = []
    avalanche_size += len(current_layer)
    
    # loop over all nodes that are in the current layer
    if len(current_layer) > 0:
        for exploding_node in current_layer:
  
            neighbours = get_neighbours(trader_dictionary[exploding_node],network)
            
            if avalanche_size == 1:
                information_per_neighbour = alpha * (trader_dictionary[exploding_node].info / (len(neighbours)))
            else:
                common_neighbours = count_common_elements(neighbours,previous_layer)
                if len(neighbours)-common_neighbours != 0:

                  
                    information_per_neighbour = alpha * (trader_dictionary[exploding_node].info / (len(neighbours)-common_neighbours))
            
            # next layer has been added to avalanche_unhandled
            
            for neighbour in neighbours:
                if neighbour not in previous_layer and neighbour not in current_layer and trader_dictionary[exploding_node].type != 'random':
                    
                    trader_dictionary[neighbour].info += information_per_neighbour

                    if trader_dictionary[neighbour].type != 'random':
                        trader_dictionary[neighbour].price = expoloding_trader_price

                    if trader_dictionary[neighbour].info >= trader_dictionary[neighbour].info_threshold and trader_dictionary[neighbour].node_number not in previous_layer and trader_dictionary[neighbour].node_number not in current_layer:
                        next_layer.append(neighbour)
                        
            trader_dictionary[exploding_node].info = 0


        current_layer = set(current_layer)
        current_layer = list(current_layer)
        previous_layer = current_layer.copy()
        next_layer = [item for item in next_layer if item not in current_layer]
        
        # recursive call to handle the next layer in the avalanche
        avalanche_size = neighbour_layer(next_layer,previous_layer, trader_dictionary, network, alpha, avalanche_size, expoloding_trader_price)

   
    return avalanche_size
    

def handle_avalanche(trader,trader_dictionary,network, alpha):

    """Handles complete avalanch from the first node that explodes after info was added to it"""

    # initializes the initial "parent" node
    avalanche_unhandled = [trader.node_number]
    expoloding_trader_price = trader.price
    avalanche_size = 0

    # call to handle current and consecutive layers originating from the "parent"
    avalanche_size = neighbour_layer(avalanche_unhandled,avalanche_unhandled, trader_dictionary, network, alpha, avalanche_size, expoloding_trader_price)
 
    return avalanche_size

def set_trader_prices(keys,trader_dictionary,global_prices,sigma, pf=5000):

    ''''Sets the trader prices at each time step depending on what type of trader it is'''

    for key in keys:
        epsilon = np.random.uniform(-sigma,sigma)
        if(trader_dictionary[key].type == 'chartist'):
            
            if(trader_dictionary[key].m >= len(global_prices)):

                pm = np.mean(global_prices)
                m_current = len(global_prices)
            else:
                pm = np.mean(global_prices[-trader_dictionary[key].m:])
                m_current = trader_dictionary[key].m

            trader_dictionary[key].decide_price(global_prices[-1], pm,epsilon, m_current)

        elif(trader_dictionary[key].type == 'fundamentalist'):
            
            trader_dictionary[key].decide_price(global_prices[-1], pf, epsilon)

        else:
            trader_dictionary[key].decide_price(global_prices[-1])

    return trader_dictionary
            
def distribute_info(trader_dictionary, network,max_info,global_prices, alpha, sigma, beta, pf, info_list):

    """Distributes info after randomly selecting node (that exceeds the threshold)"""
    
    keys = list(trader_dictionary.keys())
    
    set_trader_prices(keys,trader_dictionary,global_prices,sigma, pf)

    avalanche_counter_current_time = []
    avalanche_price_delta_list = []

    add_global_info(trader_dictionary,max_info) 

    random.shuffle(keys)
    
    # randly loop over all nodes 
    for key in keys:
        
        # if nodes are active the avalanche caused is handled
        if trader_dictionary[key].info >= trader_dictionary[key].info_threshold:
            
            avalanche_size = handle_avalanche(trader_dictionary[key],trader_dictionary,network,alpha)

            avalanche_counter_current_time.append(avalanche_size)
            avalanche_price_delta = trader_dictionary[key].price - global_prices[-1]
            avalanche_price_delta_list.append(avalanche_price_delta)

    global_prices.append(global_price_calculate(trader_dictionary, sigma, beta))
    information_list = [trader_dictionary[key].info for key in keys]

    average_info = np.mean(information_list)
    max_info = np.max(information_list)

    info_list[0].append(average_info)
    info_list[1].append(max_info)

    return avalanche_counter_current_time, avalanche_price_delta_list, global_prices, info_list

def run_simulation(network_params,network,max_info,global_prices,alpha,sigma, beta, pf, info_list,avalanches, num_days):

    '''Generates a time series as well as any other necessary outputs from the simulation'''

    if network == 'small_world':
        network, trader_dictionary = create_trader_network(network_params[0], network_params[1], network_params[2],network_params[3],network_params[4], network_params[5])
    if network == 'barabasi':
        network, trader_dictionary = create_trader_network_barabasi(network_params[0],network_params[3],network_params[4], network_params[5])                                                          
    
    # run simulation for t timesteps
    for t in range(num_days):
        if t+1 % int(num_days/10) == 0:
            clear_output(wait=True)
            print(f"Simulating timestep:{t+1}/{num_days}")
            
            
        avalanche_counter_current_time, avalanche_price_delta_list, global_prices, info_list = distribute_info(trader_dictionary, network,max_info,global_prices,alpha,sigma, beta, pf, info_list)
        avalanches.append(avalanche_counter_current_time)
    
        sum_avalanch_per_day = []

    for day in avalanches:

        sum = np.sum(day)
        sum_avalanch_per_day.append(sum)

    return avalanches,avalanche_counter_current_time, avalanche_price_delta_list, global_prices, info_list, sum_avalanch_per_day 


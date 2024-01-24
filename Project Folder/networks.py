import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import random

class Trader:

    """Defining a class for a trader object"""


    def __init__(self, node_number):
        self.node_number = node_number
        self.info_threshold = 1
        self.info = 0

def create_trader_network(num_traders, avg_degree, rewiring_probability):

    """Creates a small world network of traders"""

    traders = [Trader(i) for i in range(num_traders)]
    network = nx.watts_strogatz_graph(n=num_traders, k=avg_degree, p=rewiring_probability)

    for i, node in enumerate(network.nodes()):
        network.nodes[node]['trader'] = traders[i]

    trader_dictionary = {trader.node_number: trader for trader in traders}

    return network, trader_dictionary

def display_network(network):

    """Plots the network structure"""

    nx.draw(network, with_labels=True, node_color='lightblue', edge_color='red')
    plt.show()
    
def get_neighbours(trader,network):

    """Gets the neighbours of a given node"""


    neighbors = list(network.neighbors(trader.node_number))

    return neighbors

def add_global_info(trader_dictionary,max_info):

    """Adds global information to the network"""

    for key in trader_dictionary:
        trader_dictionary[key].info += random.uniform(0,max_info) #Max info is the maximum amount of info that can be distirbuted into the network



def distribute_info(trader_dictionary, network,max_info):

    """Distributes info after randomly selecting node (that exceeds the threshold)"""

    print("BEFORE UPDATE")
    for key in trader_dictionary:

        print(trader_dictionary[key].info)

    add_global_info(trader_dictionary,max_info)

    print("AFTER UPDATE")

    for key in trader_dictionary:
        print(trader_dictionary[key].info)

    #print(trader_dictionary.values().info)

    keys = list(trader_dictionary.keys())

    random.shuffle(keys)
    
    #for key in keys:
        
        #if trader_dictionary[key].info >= trader_dictionary[key].info_threshold:

            
            
            #neighbors = list(network.neighbors(trader.node_number))
            #total_info_to_distribute = 0.5 * trader.info
            #info_per_neighbor = total_info_to_distribute / len(neighbors)

    #         for neighbor_node in neighbors:
    #             info_to_distribute[neighbor_node] += info_per_neighbor
    #         trader.info = 0
    # for trader in traders:
    #     trader.info += info_to_distribute[trader.node_number]


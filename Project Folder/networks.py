import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import random

class Trader:
    def __init__(self, node_number):
        self.node_number = node_number
        self.info_threshold = 1
        self.info = 0

def create_trader_network(num_traders, avg_degree, rewiring_probability):
    traders = [Trader(i) for i in range(num_traders)]
    network = nx.watts_strogatz_graph(n=num_traders, k=avg_degree, p=rewiring_probability)

    for i, node in enumerate(network.nodes()):
        network.nodes[node]['trader'] = traders[i]
    return network, traders

def distribute_info(traders, network):
    info_to_distribute = {trader.node_number: 0 for trader in traders}

    for trader in traders:
        if trader.info >= trader.info_threshold:

            neighbors = list(network.neighbors(trader.node_number))
            total_info_to_distribute = 0.5 * trader.info
            info_per_neighbor = total_info_to_distribute / len(neighbors)

            for neighbor_node in neighbors:
                info_to_distribute[neighbor_node] += info_per_neighbor
            trader.info = 0
    for trader in traders:
        trader.info += info_to_distribute[trader.node_number]


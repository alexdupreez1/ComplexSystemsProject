import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import random

class Trader:
    def __init__(self, node_number):
        self.node_number = node_number
        self.info_threshold = 1
        self.info = 0

    def __repr__(self):
        return f"Trader{self.node_number}"


def create_trader_network(num_traders, avg_degree):
    traders = [Trader(i) for i in range(num_traders)]
    network = nx.random_regular_graph(d=avg_degree, n=num_traders)

    for i, node in enumerate(network.nodes()):
        network.nodes[node]['trader'] = traders[i]

    return network, traders


def distribute_info(trader, neighbors, network):
    
    if trader.info >= trader.info_threshold:
        total_info_to_distribute = 0.7 * trader.info
        info_per_neighbor = total_info_to_distribute / len(neighbors)


        for neighbor_node in neighbors:
            neighbor = network.nodes[neighbor_node]['trader']
            neighbor.info += info_per_neighbor

 
        trader.info = 0


num_traders = 200  
avg_degree = 5    

trader_network, traders = create_trader_network(num_traders, avg_degree)


fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(trader_network)  # positions for all nodes

def update(frame):
   
    for trader in traders:
        trader.info = random.uniform(0, 0.2)
        

    for trader in traders:
        neighbors = list(trader_network.neighbors(trader.node_number))
        distribute_info(trader, neighbors, trader_network)

  
    node_colors = [trader_network.nodes[node]['trader'].info for node in trader_network.nodes]

    ax.clear()
    nx.draw(trader_network, pos, node_color=node_colors, with_labels=True, cmap=plt.cm.binary, ax=ax, edge_color='gray')
    ax.set_title("Trader Network (Frame {})".format(frame))


ani = FuncAnimation(fig, update, frames=100, interval=500)

plt.show()
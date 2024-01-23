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


def create_trader_network(num_traders, avg_degree,rewiring_probability):
    traders = [Trader(i) for i in range(num_traders)]
    network = nx.watts_strogatz_graph(n=num_traders, k=avg_degree, p=rewiring_probability)

    for i, node in enumerate(network.nodes()):
        network.nodes[node]['trader'] = traders[i]

    return network, traders


def distribute_info(trader, network):
    for trader in traders:
        neighbors = list(trader_network.neighbors(trader.node_number))
        if trader.info >= trader.info_threshold:
            total_info_to_distribute = 0.5 * trader.info

            # equal or randomize?
            info_per_neighbor = total_info_to_distribute / len(neighbors)


            for neighbor_node in neighbors:
                neighbor = network.nodes[neighbor_node]['trader']
                neighbor.info += info_per_neighbor

 
            trader.info = 0


num_traders = 200  
avg_degree = 5
rewiring_probability = 0.02    

trader_network, traders = create_trader_network(num_traders, avg_degree,rewiring_probability)


fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(trader_network)  # positions for all nodes

def update(frame):
   
    for trader in traders:
        trader.info = random.uniform(0, 0.2)
        

    
    distribute_info(trader, trader_network)

  
    node_colors = [trader_network.nodes[node]['trader'].info for node in trader_network.nodes]

    ax.clear()
    nx.draw(trader_network, pos, node_color=node_colors, with_labels=True, cmap=plt.cm.binary, ax=ax, edge_color='gray')
    ax.set_title("Trader Network (Frame {})".format(frame))


ani = FuncAnimation(fig, update, frames=100, interval=500)

plt.show()
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
    initial_overflows = [trader for trader in traders if trader.info >= trader.info_threshold]
    info_queue = initial_overflows.copy()
    avalanche_sizes = []
    current_avalanche_size = 0

    while info_queue:
        current_trader = info_queue.pop(0)
        if current_trader.info < current_trader.info_threshold:
            continue

        # New avalanche detection
        if current_trader in initial_overflows:
            initial_overflows.remove(current_trader)
            if current_avalanche_size > 0:
                avalanche_sizes.append(current_avalanche_size)
            current_avalanche_size = 0

        neighbors = list(network.neighbors(current_trader.node_number))
        total_info_to_distribute = 0.5 * current_trader.info
        info_per_neighbor = total_info_to_distribute / len(neighbors)

        for neighbor_node in neighbors:
            neighbor = network.nodes[neighbor_node]['trader']
            neighbor.info += info_per_neighbor
            if neighbor.info >= neighbor.info_threshold and neighbor not in info_queue:
                info_queue.append(neighbor)
                current_avalanche_size += 1

        current_trader.info = 0

    if current_avalanche_size > 0:
        avalanche_sizes.append(current_avalanche_size)

    return avalanche_sizes


# Rest of your code remains the same
num_traders = 200  
avg_degree = 5
rewiring_probability = 0.02    
trader_network, traders = create_trader_network(num_traders, avg_degree, rewiring_probability)




fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(trader_network)

def update(frame):
    for trader in traders:
        trader.info = random.uniform(0, 0.4)

    avalanche_sizes = distribute_info(traders, trader_network)
    print(f"Frame {frame}: Avalanche Sizes = {avalanche_sizes}")


    # Distribute information
    distribute_info(traders, trader_network)

    # Normalize the information values for coloring
    max_info = max(trader.info for trader in traders)
    node_colors = [trader.info / max_info if max_info > 0 else 0 for trader in traders]

    ax.clear()
    nx.draw(trader_network, pos, node_color=node_colors, cmap=plt.cm.binary, with_labels=True, ax=ax, edge_color='gray')
    ax.set_title("Trader Network (Frame {})".format(frame))

ani = FuncAnimation(fig, update, frames=100, interval=500)
plt.show()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from networks import create_trader_network, distribute_info, Trader
import random

def run_animation():
    num_traders = 200
    avg_degree = 5
    rewiring_probability = 0.02    
    trader_network, traders = create_trader_network(num_traders, avg_degree, rewiring_probability)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(trader_network)

    def update(frame):
        for trader in traders:
            trader.info = random.uniform(0, 0.2)
        distribute_info(traders, trader_network)
        max_info = max(trader.info for trader in traders)
        node_colors = [trader.info / max_info if max_info > 0 else 0 for trader in traders]
        ax.clear()
        nx.draw(trader_network, pos, node_color=node_colors, cmap=plt.cm.viridis, with_labels=True, ax=ax, edge_color='gray')
        ax.set_title("Trader Network (Frame {})".format(frame))

    ani = FuncAnimation(fig, update, frames=100, interval=500)
    plt.show()

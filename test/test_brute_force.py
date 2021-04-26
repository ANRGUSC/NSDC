from bnet.generators import random_task_graph, random_network
from bnet.optimizer import find_network
from itertools import combinations
from networkx import nx

def main():
    total_nodes = 10
    max_nodes = 5
    task_order = 5

    complete_network = random_network(total_nodes, comp_multiplier_range=(3, 10), comm_range=(2, 5))
    makespan, network = find_network(
        networks=[
            nx.subgraph(complete_network, sub)
            for sub in combinations(complete_network.nodes, r=max_nodes)
        ],
        task_graph_sampler=lambda: random_task_graph(task_order, comp_range=[1, 4], data_range=[0, 3]),
        samples=10
    )

    print(f"Best Network (Makespan {makespan}): {list(network.nodes)}")

if __name__ == "__main__":
    main()
from bnet import TaskGraph, Network
from bnet.optimizers import BruteForceOptimizer
from bnet.generators import NaiveNetworkGenerator, NaiveTaskGraphGenerator
from bnet.schedulers import HeftScheduler

def main():
    # Get Networks 
    num_networks = 5
    network_generator = NaiveNetworkGenerator(
        num_nodes=5, 
        comp_multiplier_range=[1, 5],
        comm_range=[1, 10]
    )
    task_graph_generator = NaiveTaskGraphGenerator(
        num_tasks=10,
        comp_range=[1, 5],
        data_range=[1, 4]
    )

    networks = [network_generator.generate() for _ in range(num_networks)]

    optimizer = BruteForceOptimizer(samples=10)
    for network, score, task_graph, best_network, best_score in optimizer.optimize_iter(
            networks=networks,
            task_graph_generator=task_graph_generator,
            scheduler=HeftScheduler()):
        print(score, best_score)
    print(best_score)


if __name__ == "__main__":
    main()
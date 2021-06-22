from bnet.task_graph import SimpleTaskGraph
from bnet.network import SimpleNetwork
from bnet.optimizers import BruteForceOptimizer
from bnet.schedulers import HeftScheduler

def main():
    task_graph_generator = SimpleTaskGraph.random_generator(
        num_tasks=10,
        task_cost={
            SimpleTaskGraph.Cost.LOW: 300,
            SimpleTaskGraph.Cost.HIGH: 500,
        },
        data_cost={
            SimpleTaskGraph.Cost.LOW: 300,
            SimpleTaskGraph.Cost.HIGH: 500,
        }
    )

    # Generate a "mother network" and assign values to high/low values
    network = SimpleNetwork.random(
        num_nodes=5,
        node_speed={
            SimpleNetwork.Speed.LOW: 50,
            SimpleNetwork.Speed.HIGH: 100,
        },
        sat_speed={
            SimpleNetwork.Speed.LOW: 5,
            SimpleNetwork.Speed.HIGH: 10,
        },
        radio_speed={
            SimpleNetwork.Speed.LOW: 2,
            SimpleNetwork.Speed.HIGH: 4,
        },
        gray_speed={
            SimpleNetwork.Speed.LOW: 10,
            SimpleNetwork.Speed.HIGH: 20,
        }
    )

    optimizer = BruteForceOptimizer(samples=10)
    for network, score, task_graph, best_network, best_score in optimizer.optimize_iter(
            networks=network.iter_subnetworks(),
            task_graph_generator=task_graph_generator,
            scheduler=HeftScheduler()):
        print(score, best_score)
    print(best_score)


if __name__ == "__main__":
    main()
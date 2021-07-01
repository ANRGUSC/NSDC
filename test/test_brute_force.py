from bnet.task_graph.eugenio_simple_task_graph_generator import EugenioSimpleTaskGraphGenerator
from bnet.task_graph import SimpleTaskGraph
from bnet.network import SimpleNetwork
from bnet.optimizers import BruteForceOptimizer
from bnet.schedulers import HeftScheduler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.patches as mpatches


def main():
    # Generate a "mother network" and assign values to high/low values
    mother_network = SimpleNetwork.random(
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

    # Create a task graph generator so the optimizer can sample task graphs
    # task_graph_generator = SimpleTaskGraph.random_generator(
    #     num_tasks=10,
    #     task_cost={
    #         SimpleTaskGraph.Cost.LOW: 300,
    #         SimpleTaskGraph.Cost.HIGH: 500,
    #     },
    #     data_cost={
    #         SimpleTaskGraph.Cost.LOW: 300,
    #         SimpleTaskGraph.Cost.HIGH: 500,
    #     }
    # )

    task_graph_generator = EugenioSimpleTaskGraphGenerator(10, 15)

    results = BruteForceOptimizer(samples=10).optimize_iter(
        networks=mother_network.iter_subnetworks(),     # optimize over all subnetworks of the mother network
        task_graph_generator=task_graph_generator,      # task graph generater to generate task graphs for scheduler
        scheduler=HeftScheduler()                       # scheduler to optimize with
    )



    # Visualization stuff - Not necessary for optimization
    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10))

    cur_network, scores = None, []
    def update(frame):
        nonlocal mother_network, ax1, ax2, ax3, cur_network, scores #, queue
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.set_title("Current Subnetwork")
        ax2.set_title("Sampled Task Graph")
        ax3.set_title("Best Subnetwork Found")
        
        ax1.legend(
            handles=[
                mpatches.Patch(color='#A3BE8C', label='High Speed Connection/Satellite'),
                mpatches.Patch(color='#88C0D0', label='Low Speed Connection/Satellite'),
                mpatches.Patch(color='gray', label='No Satellite Connection')
            ]
        )

        ax2.legend(
            handles=[
                mpatches.Patch(color='#BF616A', label='High Cost/Data'),
                mpatches.Patch(color='#EBCB8B', label='Low Cost/Data')
            ]
        )

        task_graph: SimpleTaskGraph
        network, score, task_graph, best_network, best_score = next(results)
        if network != cur_network:
            cur_network, scores = network, []
            
        scores.append(score)
        ax1.set_xlabel(f"Average Score: {np.mean(scores):.2f} ({len(scores)} samples)")
        ax3.set_xlabel(f"Average Score: {best_score:.2f}")

        mother_network.draw(network.edges, ax=ax1)
        if best_network is not None:
            mother_network.draw(best_network.edges, ax=ax3)
        if task_graph is not None:
            task_graph.draw(ax=ax2)

        
    ani = FuncAnimation(fig, update, interval=0)
    plt.show()



if __name__ == "__main__":
    main()
from bnet.task_graph.eugenio_simple_task_graph_generator import EugenioSimpleTaskGraphGenerator
from bnet.task_graph import SimpleTaskGraph
from bnet.network import SimpleNetwork
from bnet.optimizers import BruteForceOptimizer
from bnet.schedulers import HeftScheduler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


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
    #     num_tasks=15,
    #     task_cost={
    #         SimpleTaskGraph.Cost.LOW: 300,
    #         SimpleTaskGraph.Cost.HIGH: 500,
    #     },
    #     data_cost={
    #         SimpleTaskGraph.Cost.LOW: 300,
    #         SimpleTaskGraph.Cost.HIGH: 500,
    #     }
    # )

    task_graph_generator = EugenioSimpleTaskGraphGenerator(
        nodes=15, 
        edges=25,
        task_cost={
            SimpleTaskGraph.Cost.LOW: 300,
            SimpleTaskGraph.Cost.HIGH: 500,
        },
        data_cost={
            SimpleTaskGraph.Cost.LOW: 300,
            SimpleTaskGraph.Cost.HIGH: 500,
        }
    )

    results = BruteForceOptimizer(samples=5).optimize_iter(
        networks=mother_network.iter_subnetworks(),     # optimize over all subnetworks of the mother network
        task_graph_generator=task_graph_generator,      # task graph generater to generate task graphs for scheduler
        scheduler=HeftScheduler()                       # scheduler to optimize with
    )

    # Visualization stuff - Not necessary for optimization
    ax_cur_network: plt.Axes
    ax_cur_dag: plt.Axes
    ax_best_network: plt.Axes
    ax_score: plt.Axes

    fig = plt.figure(figsize=(17, 9))
    fig.suptitle("Network Optimization")

    gs = GridSpec(nrows=3, ncols=3, hspace=0.4, wspace=0.1) #, figure=fig)
    ax_score = fig.add_subplot(gs[-1,:])
    ax_cur_network = fig.add_subplot(gs[:-1,0])
    ax_cur_dag = fig.add_subplot(gs[:-1,1])
    ax_best_network = fig.add_subplot(gs[:-1,2])

    scatter = ax_score.scatter([], [])

    cur_network, avg_scores, scores, costs = None, [], [], []
    def update(frame):
        nonlocal mother_network, ax_cur_network, ax_cur_dag, ax_best_network, cur_network, scores #, queue
        
        task_graph: SimpleTaskGraph
        try:
            network, score, task_graph, best_network, best_score = next(results)
        except StopIteration:
            return 

        if network != cur_network:
            if scores and cur_network is not None:
                costs.append(cur_network.cost())
                avg_scores.append(np.mean(scores))

                scatter.set_offsets(list(zip(costs, avg_scores)))
                ax_score.set_xlim(min(costs)*0.9, max(costs)*1.1)
                ax_score.set_ylim(min(avg_scores)*0.9, max(avg_scores)*1.1)

            cur_network, scores = network, []
        
        ax_cur_network.clear()
        ax_cur_dag.clear()
        ax_best_network.clear()
        ax_cur_network.set_title("Current Subnetwork")
        ax_cur_dag.set_title("Sampled Task Graph")
        ax_best_network.set_title("Best Subnetwork Found")
        ax_score.set_title("Cost-Score Results")
        ax_score.set_xlabel("Cost")
        ax_score.set_ylabel("Score")
        
        ax_cur_network.legend(
            handles=[
                mpatches.Patch(color='#A3BE8C', label='High Speed Connection/Satellite'),
                mpatches.Patch(color='#88C0D0', label='Low Speed Connection/Satellite'),
                mpatches.Patch(color='gray', label='No Satellite Connection')
            ]
        )

        ax_cur_dag.legend(
            handles=[
                mpatches.Patch(color='#BF616A', label='High Cost/Data'),
                mpatches.Patch(color='#EBCB8B', label='Low Cost/Data')
            ],
            loc="upper left"
        )

        scores.append(score)

        ax_cur_network.set_xlabel(f"Average Score: {np.mean(scores):.2f} ({len(scores)} samples)")
        ax_best_network.set_xlabel(f"Average Score: {best_score:.2f}")

        mother_network.draw(network.edges, ax=ax_cur_network)
        if best_network is not None:
            mother_network.draw(best_network.edges, ax=ax_best_network)
        if task_graph is not None:
            task_graph.draw(ax=ax_cur_dag)

        
    ani = FuncAnimation(fig, update, interval=0)
    plt.show()



if __name__ == "__main__":
    main()
from matplotlib.lines import Line2D
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
import matplotlib.cm as cm
from bnet.generators.generator import CycleGenerator

import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples", type=int, required=True)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    ## MOTHER NETWORK ##
    # Construct Network
    mother_network = SimpleNetwork(
        node_speed={
            SimpleNetwork.Speed.LOW: 10,        # Node speed, a node with speed s computes a task with size t in n/s seconds
            SimpleNetwork.Speed.HIGH: 100,
        },
        sat_speed={
            SimpleNetwork.Speed.LOW: 5,         # Satellite speed, an edge with speed (trasnfer rate) s transmits data of size d between two nodes in s/d seconds 
            SimpleNetwork.Speed.HIGH: 10,
        },
        radio_speed={
            SimpleNetwork.Speed.LOW: 2,
            SimpleNetwork.Speed.HIGH: 4,
        },
        gray_speed={
            SimpleNetwork.Speed.LOW: 100,
            SimpleNetwork.Speed.HIGH: 200,
        }
    )
    # Add 5 nodes with specified compute speeds and positions
    mother_network.add_node(0, SimpleNetwork.Speed.LOW, pos=(0, 0.25))
    mother_network.add_node(1, SimpleNetwork.Speed.LOW, pos=(0.35, 0.3))
    mother_network.add_node(2, SimpleNetwork.Speed.HIGH, pos=(0.4, 0.15))
    mother_network.add_node(3, SimpleNetwork.Speed.HIGH, pos=(0.8, 0.3))
    mother_network.add_node(4, SimpleNetwork.Speed.HIGH, pos=(0.6, 0.6))

    # Add gray network edges
    mother_network.add_gray_edge(2, 3, SimpleNetwork.Speed.HIGH)
    mother_network.add_gray_edge(3, 4, SimpleNetwork.Speed.HIGH)
    mother_network.add_gray_edge(2, 4, SimpleNetwork.Speed.LOW)

    # Add radio network edges
    mother_network.add_radio_edge(1, 0, SimpleNetwork.Speed.HIGH)
    mother_network.add_radio_edge(2, 1, SimpleNetwork.Speed.HIGH)
    mother_network.add_radio_edge(2, 0, SimpleNetwork.Speed.LOW)
    mother_network.add_radio_edge(2, 3, SimpleNetwork.Speed.LOW)
    mother_network.add_radio_edge(3, 4, SimpleNetwork.Speed.LOW)
    mother_network.add_radio_edge(1, 4, SimpleNetwork.Speed.LOW)

    # Add satellite edges
    mother_network.add_satellite_edge(1, SimpleNetwork.Speed.LOW)
    mother_network.add_satellite_edge(4, SimpleNetwork.Speed.HIGH)
    mother_network.add_satellite_edge(2, SimpleNetwork.Speed.HIGH)



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

    _task_graph_generator = EugenioSimpleTaskGraphGenerator(
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

    task_graph_generator = CycleGenerator(
        task_graphs=[
            _task_graph_generator.generate()
            for _ in range(args.samples)
        ]
    )

    scheduler = HeftScheduler()
    results = BruteForceOptimizer(
        samples=args.samples,
        networks=mother_network.iter_subnetworks(),     # optimize over all subnetworks of the mother network
        task_graph_generator=task_graph_generator,      # task graph generater to generate task graphs for scheduler
        cost_func=lambda makespan, deploy_cost, risk: deploy_cost + 20 * risk + makespan / 10,
        makespan=scheduler.schedule,
        deploy_cost=lambda network, _: network.cost(),
        risk=lambda network, _: network.risk()
    ).optimize_iter()

    # Visualization stuff - Not necessary for optimization
    ax_cur_network: plt.Axes
    ax_cur_dag: plt.Axes
    ax_best_network: plt.Axes
    ax_makespan: plt.Axes
    ax_no_risk: plt.Axes
    ax_any_risk: plt.Axes

    fig = plt.figure(figsize=(17, 9))
    plt.set_cmap('coolwarm')
    fig.suptitle(f"Network Optimization (Brute Force)")
 
    gs = GridSpec(nrows=2, ncols=3, hspace=0.4, wspace=0.1) #, figure=fig)
    ax_cur_network = fig.add_subplot(gs[:-1,0])
    ax_cur_dag = fig.add_subplot(gs[:-1,1])
    ax_best_network = fig.add_subplot(gs[:-1,2])
    ax_makespan = fig.add_subplot(gs[-1:,0])
    ax_no_risk = fig.add_subplot(gs[-1:,1])
    ax_any_risk = fig.add_subplot(gs[-1:,2])

    scatter = ax_makespan.scatter([], []) 
    cbar = fig.colorbar(scatter, ax=ax_makespan, ticks=[0, 1])
    cbar.ax.set_yticklabels(["low risk", "high risk"])

    max_risk = sum(key=="gray" for _, _, key in mother_network.edges)

    cur_network, cur_metrics, avg_makespans, makespans, deploy_costs, risks, costs = None, {}, [], [], [], [], []
    best_network, best_metrics = None, {}
    best_any_risk_network, best_any_risk_metrics = None, {}
    best_no_risk_network, best_no_risk_metrics = None, {}
    def update(frame):
        nonlocal mother_network, ax_cur_network, ax_cur_dag, ax_best_network
        nonlocal cur_network, cur_metrics, makespans, best_metrics, best_network
        nonlocal best_any_risk_network, best_any_risk_metrics
        nonlocal best_no_risk_network, best_no_risk_metrics
        
        task_graph: SimpleTaskGraph
        try:
            network, _metrics, task_graph, _, _ = next(results)
        except StopIteration:
            return 

        if network != cur_network:
            if makespans and cur_network is not None:
                deploy_costs.append(cur_metrics["deploy_cost"])
                risks.append(cur_metrics["risk"])
                avg_makespans.append(np.mean(makespans))
                costs.append(cur_metrics["cost"])
                cur_metrics["avg_makespan"] = avg_makespans[-1]

                if costs[-1] < best_metrics.get("cost", np.inf):
                    best_network, best_metrics = cur_network, cur_metrics
                if risks[-1] == 0:
                    if avg_makespans[-1] < best_no_risk_metrics.get("avg_makespan", np.inf):
                        best_no_risk_network, best_no_risk_metrics = cur_network, cur_metrics
                if avg_makespans[-1] < best_any_risk_metrics.get("avg_makespan", np.inf):
                    best_any_risk_network, best_any_risk_metrics = cur_network, cur_metrics

                scatter.set_offsets(list(zip(deploy_costs, np.log10(avg_makespans))))
                scatter.set_edgecolors(cm.coolwarm(np.array(risks) / max_risk))
                scatter.set_facecolors(cm.coolwarm(np.array(risks) / max_risk))
                ax_makespan.set_xlim(min(deploy_costs)*0.9, max(deploy_costs)*1.1)
                ax_makespan.set_ylim(min(np.log10(avg_makespans))*0.9, max(np.log10(avg_makespans))*1.1)

            cur_network, cur_metrics, makespans = network, _metrics, []

        
        ax_cur_network.clear()
        ax_cur_dag.clear()
        ax_best_network.clear()
        ax_no_risk.clear()
        ax_any_risk.clear()
        
        ax_cur_network.set_title("Current Subnetwork")
        ax_cur_dag.set_title("Sampled Task Graph")
        ax_best_network.set_title("Best Subnetwork Found (Cost Function)")
        ax_any_risk.set_title("Best Subnetwork Found (Any Deploy-Cost/Risk)")
        ax_no_risk.set_title("Best Subnetwork Found (No Risk/Any Deploy-Cost)")
        ax_makespan.set_title("Deploy-Cost/Makespan/Risk Results")
        ax_makespan.set_xlabel("Deploy-Cost")
        ax_makespan.set_ylabel("Makespan (log-scale)")
        
        ax_cur_network.legend(handles=[  
            Line2D( # nosat
                [0], [0], marker='o', color='w', label='No Satellite',
                markerfacecolor='#BF616A', markersize=10
            ),
            Line2D( # low sat
                [0], [0], marker='o', color='w', label='Low Speed Satellite',
                markerfacecolor='#88C0D0', markersize=10
            ),
            Line2D( # high sat
                [0], [0], marker='o', color='w', label='High Speed Satellite',
                markerfacecolor='#A3BE8C', markersize=10
            ),
            Line2D( # low radio
                [0], [0],  color='#88C0D0', linestyle="dashed", 
                label='Low Speed Radio'
            ),
            Line2D( # high radio
                [0], [0], color='#A3BE8C', linestyle="dashed",
                label='High Speed Radio'
            ),
            Line2D( # low gray
                [0], [0],  color='#88C0D0', label='Low Speed Gray Cellular'
            ),
            Line2D( # high gray
                [0], [0], color='#A3BE8C', label='High Speed Gray Cellular'
            )
        ])

        ax_cur_dag.legend(
            handles=[
                mpatches.Patch(color='#BF616A', label='High Cost/Data'),
                mpatches.Patch(color='#EBCB8B', label='Low Cost/Data')
            ],
            loc="upper left"
        )

        makespans.append(_metrics["makespan"])

        ax_cur_network.set_xlabel(f"Average makespan: {np.mean(makespans):.2f} ({len(makespans)} samples)")
        ax_best_network.set_xlabel(
            f"Deploy Cost: {best_metrics.get('deploy_cost', np.inf):.2f}, " +
            f"Risk: {best_metrics.get('risk', np.inf):.2f}, " +
            f"Makepsan: {best_metrics.get('avg_makespan', np.inf):.2f}" +
            f"\nCost: {best_metrics.get('cost', np.inf):.2f}"
        )
        ax_no_risk.set_xlabel(
            f"Deploy Cost: {best_no_risk_metrics.get('deploy_cost', np.inf):.2f}, " +
            f"Risk: {best_no_risk_metrics.get('risk', np.inf):.2f}, " +
            f"Makepsan: {best_no_risk_metrics.get('avg_makespan', np.inf):.2f}" +
            f"\nCost: {best_no_risk_metrics.get('cost', np.inf):.2f}"
        )
        ax_any_risk.set_xlabel(
            f"Deploy Cost: {best_any_risk_metrics.get('deploy_cost', np.inf):.2f}, " +
            f"Risk: {best_any_risk_metrics.get('risk', np.inf):.2f}, " +
            f"Makepsan: {best_any_risk_metrics.get('avg_makespan', np.inf):.2f}" +
            f"\nCost: {best_any_risk_metrics.get('cost', np.inf):.2f}"
        )

        mother_network.draw(network.edges, ax=ax_cur_network)
        if best_network is not None:
            mother_network.draw(best_network.edges, ax=ax_best_network)
        if task_graph is not None:
            task_graph.draw(ax=ax_cur_dag)
        if best_any_risk_network is not None:
            mother_network.draw(best_any_risk_network.edges, ax=ax_any_risk)
        if best_no_risk_network is not None:
            mother_network.draw(best_no_risk_network.edges, ax=ax_no_risk)

        
    ani = FuncAnimation(fig, update, interval=0)
    plt.show()



if __name__ == "__main__":
    main()
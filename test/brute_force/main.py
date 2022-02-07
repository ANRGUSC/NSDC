from copy import deepcopy
import pathlib
from typing import Optional
from bnet.task_graph.task_graph import TaskGraph

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from bnet.generators.generator import TaskGraphGenerator, TaskGraphSetGenerator
from bnet.network import SimpleNetwork
from bnet.optimizers.brute_force import BruteForceOptimizer
from bnet.optimizers.optimizer import Result
from bnet.schedulers import HeftScheduler
from bnet.task_graph import SimpleTaskGraph
from matplotlib.lines import Line2D

from wfcommons.wfchef.recipes import CyclesRecipe
from heft.gantt import showGanttChart

thisdir = pathlib.Path(__file__).resolve().parent
savedir = thisdir.joinpath("outputs")
savedir.mkdir(parents=True, exist_ok=True)

NUM_TASK_GRAPHS = 1
COEF_MAKESPAN = 1/50
COEF_RISK = 1
COEF_DEPLOY_COST = 1.25


def get_network() -> SimpleNetwork:
    ## MOTHER NETWORK ##
    # Construct Network
    mother_network = SimpleNetwork(
        node_speed={
            # Node speed, a node with speed s computes a task with size t
            # in n/s seconds
            SimpleNetwork.Speed.LOW: 1,
            SimpleNetwork.Speed.HIGH: 2,
        },
        sat_speed={
            # Satellite speed, an edge with speed (transfer rate) s transmits
            # data of size d between two nodes in s/d seconds
            SimpleNetwork.Speed.LOW: 3,
            SimpleNetwork.Speed.HIGH: 4,
        },
        radio_speed={
            SimpleNetwork.Speed.LOW: 1,
            SimpleNetwork.Speed.HIGH: 2,
        },
        gray_speed={
            SimpleNetwork.Speed.LOW: 10,
            SimpleNetwork.Speed.HIGH: 20,
        },

        # Deploy Cost by resource
        node_cost={
            SimpleNetwork.Speed.LOW: 1,
            SimpleNetwork.Speed.HIGH: 2,
        },
        sat_cost={
            SimpleNetwork.Speed.LOW: 7,
            SimpleNetwork.Speed.HIGH: 10,
        },
        radio_cost={
            SimpleNetwork.Speed.LOW: 6,
            SimpleNetwork.Speed.HIGH: 8,
        },
        gray_cost={
            SimpleNetwork.Speed.LOW: 1,
            SimpleNetwork.Speed.HIGH: 2,
        }
    )
    # Add 5 nodes with specified compute speeds and positions
    mother_network.add_node(0, SimpleNetwork.Speed.LOW, pos=(0, 0.25))
    mother_network.add_node(1, SimpleNetwork.Speed.LOW, pos=(0.35, 0.3))
    mother_network.add_node(2, SimpleNetwork.Speed.LOW, pos=(0.4, 0.15))
    mother_network.add_node(3, SimpleNetwork.Speed.HIGH, pos=(0.8, 0.3))
    mother_network.add_node(4, SimpleNetwork.Speed.HIGH, pos=(0.6, 0.6))

    # Add gray network edges
    mother_network.add_gray_edge(2, 3, SimpleNetwork.Speed.HIGH)
    mother_network.add_gray_edge(3, 4, SimpleNetwork.Speed.HIGH)
    mother_network.add_gray_edge(2, 4, SimpleNetwork.Speed.LOW)

    # Add radio network edges
    mother_network.add_radio_edge(1, 0, SimpleNetwork.Speed.LOW)
    mother_network.add_radio_edge(2, 1, SimpleNetwork.Speed.HIGH)
    mother_network.add_radio_edge(2, 0, SimpleNetwork.Speed.LOW)
    mother_network.add_radio_edge(2, 3, SimpleNetwork.Speed.LOW)

    # Add satellite edges
    mother_network.add_satellite_edge(1, SimpleNetwork.Speed.LOW)
    mother_network.add_satellite_edge(4, SimpleNetwork.Speed.LOW)
    # mother_network.add_satellite_edge(2, SimpleNetwork.Speed.HIGH)

    return mother_network

class CyclesGenerator(TaskGraphGenerator):
    def __init__(self, num_tasks: int) -> None:
        super().__init__()
        self.num_tasks = num_tasks

        # recipe = CyclesRecipe(num_tasks=self.num_tasks)
        # graph = recipe.generate_nx_graph()

        self.task_costs = {
            "SRC": SimpleTaskGraph.Cost.NONE,
            "DST": SimpleTaskGraph.Cost.NONE,
            "baseline_cycles": SimpleTaskGraph.Cost.LOW,
            "cycles_fertilizer_increase_output_parser": SimpleTaskGraph.Cost.HIGH,
            "fertilizer_increase_cycles": SimpleTaskGraph.Cost.HIGH,
            "cycles": SimpleTaskGraph.Cost.HIGH,
            "cycles_plots": SimpleTaskGraph.Cost.LOW,
            "cycles_fertilizer_increase_output_summary": SimpleTaskGraph.Cost.LOW,
            "cycles_output_summary": SimpleTaskGraph.Cost.LOW,
        }

        self.edge_costs = {
            ("SRC", "baseline_cycles"): SimpleTaskGraph.Cost.NONE,
            ("baseline_cycles", "cycles"): SimpleTaskGraph.Cost.LOW,
            ("baseline_cycles", "fertilizer_increase_cycles"): SimpleTaskGraph.Cost.HIGH,
            ("fertilizer_increase_cycles", "cycles_fertilizer_increase_output_parser"): SimpleTaskGraph.Cost.LOW,
            ("cycles", "cycles_output_summary"): SimpleTaskGraph.Cost.HIGH,
            ('cycles', 'cycles_fertilizer_increase_output_parser'): SimpleTaskGraph.Cost.HIGH,
            ("cycles_fertilizer_increase_output_parser", "cycles_output_summary"): SimpleTaskGraph.Cost.LOW,
            ("cycles_output_summary", "cycles_plots"): SimpleTaskGraph.Cost.HIGH,
            ("cycles_fertilizer_increase_output_parser", "cycles_fertilizer_increase_output_summary"): SimpleTaskGraph.Cost.LOW,
            ("cycles_plots", "DST"): SimpleTaskGraph.Cost.NONE,
            ('cycles_fertilizer_increase_output_summary', 'DST'): SimpleTaskGraph.Cost.NONE
        }
    
    def generate(self) -> SimpleTaskGraph:
        recipe = CyclesRecipe(num_tasks=self.num_tasks)
        graph = recipe.generate_nx_graph()

        task_graph = SimpleTaskGraph(
            task_cost={
                SimpleTaskGraph.Cost.LOW: 50,
                SimpleTaskGraph.Cost.HIGH: 100,
            },
            data_cost={
                SimpleTaskGraph.Cost.LOW: 5,
                SimpleTaskGraph.Cost.HIGH: 7,
            }
        )
        
        for node in graph.nodes:
            task_graph.add_task(node, self.task_costs[graph.nodes[node]["type"]])
            
        for src, dst in graph.edges:
            src_type = graph.nodes[src]["type"]
            dst_type = graph.nodes[dst]["type"]
            task_graph.add_dependency(src, dst, self.edge_costs[(src_type, dst_type)])

        return task_graph 


class ExampleGenerator(TaskGraphGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.task_graph = SimpleTaskGraph(
            task_cost={
                SimpleTaskGraph.Cost.LOW: 10,
                SimpleTaskGraph.Cost.HIGH: 50,
            },
            data_cost={
                SimpleTaskGraph.Cost.LOW: 5,
                SimpleTaskGraph.Cost.HIGH: 8,
            }
        )
        self.task_graph.add_task(1, SimpleTaskGraph.Cost.HIGH)
        self.task_graph.add_task(2, SimpleTaskGraph.Cost.LOW)
        self.task_graph.add_task(3, SimpleTaskGraph.Cost.HIGH)
        self.task_graph.add_task(4, SimpleTaskGraph.Cost.LOW)
        self.task_graph.add_task(5, SimpleTaskGraph.Cost.HIGH)
        self.task_graph.add_task(6, SimpleTaskGraph.Cost.LOW)

        self.task_graph.add_dependency(1, 2, SimpleTaskGraph.Cost.HIGH)
        self.task_graph.add_dependency(1, 3, SimpleTaskGraph.Cost.LOW)
        self.task_graph.add_dependency(1, 4, SimpleTaskGraph.Cost.HIGH)
        self.task_graph.add_dependency(4, 5, SimpleTaskGraph.Cost.HIGH)
        self.task_graph.add_dependency(3, 5, SimpleTaskGraph.Cost.LOW)
        self.task_graph.add_dependency(3, 6, SimpleTaskGraph.Cost.LOW)
        self.task_graph.add_dependency(5, 6, SimpleTaskGraph.Cost.LOW)
        self.task_graph.add_dependency(2, 6, SimpleTaskGraph.Cost.LOW)

    def generate(self) -> SimpleTaskGraph:
        return self.task_graph
    
def draw_network(path: pathlib.Path,
                 network: SimpleNetwork, 
                 subnetwork: Optional[SimpleNetwork] = None,
                 title: str = "Network") -> None:
    fig, ax = network.draw(subnetwork or network)
    ax.set_title(title)
    
    # ax_cur_network.set_xlabel(f"Average makespan: {np.mean(makespans):.2f} ({len(makespans)} samples)")
    ax.legend(handles=[  
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
    ax.axis('off')
    ax.margins(0)
    fig.savefig(path)

def draw_task_graph(path: pathlib.Path, 
                    task_graph: SimpleTaskGraph,
                    title: str = "Task Graph") -> None:
    fig, ax = task_graph.draw()
    ax.set_title(title)
    ax.legend(
        handles=[
            mpatches.Patch(color='#BF616A', label='High Cost/Data'),
            mpatches.Patch(color='#EBCB8B', label='Low Cost/Data')
        ],
        loc="upper left"
    )
    ax.axis('off')
    ax.margins(0)
    fig.savefig(path)

def main():
    mother_network = get_network()    
    draw_network(
        path=savedir.joinpath("mother_network.png"),
        network=mother_network,
        title="Mother Network"
    )

    _generator = CyclesGenerator(70)
    # _generator = ExampleGenerator()
    task_graph_generator = TaskGraphSetGenerator(
        task_graphs=[
            _generator.generate()
            for _ in range(NUM_TASK_GRAPHS)
        ]
    )
    for i in range(NUM_TASK_GRAPHS):
        task_graph: SimpleTaskGraph = task_graph_generator.generate()
        draw_task_graph(
            path=savedir.joinpath(f"task_graph_{i+1}.png"),
            task_graph=task_graph,
            title=f"Task Graph {i+1}"
        )   

    scheduler = HeftScheduler()
    proc_sched, task_sched, dict_sched  = scheduler.get_schedule(
        network=mother_network,
        task_graph=task_graph
    )

    showGanttChart(proc_sched, savedir.joinpath("mother_network_schedule.png"))

    def cost_func(network: SimpleNetwork) -> Result:
        task_graphs = []
        makespans = []
        for _ in range(NUM_TASK_GRAPHS):
            task_graphs.append(task_graph_generator.generate())
            try:
                makespan = scheduler.schedule(
                    network=network,
                    task_graph=task_graphs[-1]
                )
            except AssertionError:
                continue
            makespans.append(makespan)

        makespan = np.inf if not makespans else sum(makespans) / len(makespans)
        deploy_cost = network.cost()
        risk = network.risk()
        return Result(
            network=network,
            cost=(
                COEF_DEPLOY_COST * deploy_cost + 
                COEF_RISK * risk +
                COEF_MAKESPAN * makespan
            ),
            metadata={
                "task_graphs": task_graphs,
                "makespans": makespans,
                "deploy_cost": deploy_cost,
                "risk": risk,
            }
        )

    optimizer = BruteForceOptimizer(
        networks=mother_network.iter_subnetworks(),
        cost_func=cost_func
    )

    columns=["avg_makespan", "deploy_cost", "risk", "cost"]
    rows = []
    best_cost: float = np.inf
    best_network: Optional[SimpleNetwork] = None 
    for res in optimizer.optimize_iter():
        if res.cost > 100000:
            continue
        rows.append([
            np.inf if not res.metadata["makespans"] else np.mean(res.metadata["makespans"]),
            res.metadata["deploy_cost"],
            res.metadata["risk"],
            res.cost,
        ])
        print(dict(zip(columns, rows[-1])))

        if res.cost < best_cost:
            best_cost, best_network = res.cost, deepcopy(res.network)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(savedir.joinpath("results.csv"))

    draw_network(
        path=savedir.joinpath("best_network.png"),
        network=mother_network,
        subnetwork=best_network,
        title="Best Subnetwork"
    )

    task_graph = task_graph_generator.generate()
    proc_sched, task_sched, dict_sched  = scheduler.get_schedule(
        network=best_network,
        task_graph=task_graph
    )
    
    showGanttChart(proc_sched, savedir.joinpath("best_network_schedule.png"))


if __name__ == "__main__":
    main()

import pathlib
import random
from typing import List, Optional
import matplotlib
from nsdc.optimizers.simulated_annealing import ExponentialCool, SimulatedAnnealingOptimizer
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from nsdc.generators.generator import TaskGraphGenerator, TaskGraphSetGenerator
from nsdc.network import SimpleNetwork
from nsdc.optimizers.exhaustive import ExhaustiveSearch
from nsdc.optimizers.optimizer import Result
from nsdc.schedulers import HeftScheduler
from nsdc.task_graph import SimpleTaskGraph
from matplotlib.lines import Line2D

from heft.gantt import showGanttChart
import warnings
warnings.filterwarnings('ignore')

def saveGanttChart(proc_schedules, path: pathlib.Path):
    showGanttChart(proc_schedules)
    plt.savefig(path)
    plt.close()

thisdir = pathlib.Path(__file__).resolve().parent
savedir = thisdir.joinpath("outputs")
savedir.mkdir(parents=True, exist_ok=True)

NUM_TASK_GRAPHS = 5
COEF_MAKESPAN = 1/10
COEF_RISK = 1
COEF_DEPLOY_COST = 1/10

OUTLIER_COST = 100
MAX_ITERATIONS = 100


def get_network() -> SimpleNetwork:
    ## MOTHER NETWORK ##
    # Construct Network
    mother_network = SimpleNetwork(
        node_speed={
            # Node speed, a node with speed s computes a task with size t
            # in n/s seconds
            SimpleNetwork.Speed.LOW: 1/10,
            SimpleNetwork.Speed.HIGH: 2/10,
        },
        sat_speed={
            # Satellite speed, an edge with speed (transfer rate) s transmits
            # data of size d between two nodes in s/d seconds
            SimpleNetwork.Speed.LOW: 1/5,
            SimpleNetwork.Speed.HIGH: 2/5,
        },
        radio_speed={
            SimpleNetwork.Speed.LOW: 1/5,
            SimpleNetwork.Speed.HIGH: 2/5,
        },
        gray_speed={
            SimpleNetwork.Speed.LOW: 4/5,
            SimpleNetwork.Speed.HIGH: 8/5,
        },

        # Deploy Cost by resource
        node_cost={
            SimpleNetwork.Speed.LOW: 1,
            SimpleNetwork.Speed.HIGH: 2,
        },
        sat_cost={
            SimpleNetwork.Speed.LOW: 5,
            SimpleNetwork.Speed.HIGH: 10,
        },
        radio_cost={
            SimpleNetwork.Speed.LOW: 5,
            SimpleNetwork.Speed.HIGH: 10,
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

class ExampleGenerator(TaskGraphGenerator):
    def __init__(self, levels: List[int]) -> None:
        super().__init__()
        self.levels = levels

    def generate(self) -> SimpleTaskGraph:
        choices = [
            SimpleTaskGraph.Cost.LOW,
            SimpleTaskGraph.Cost.HIGH
        ]
        task_graph = SimpleTaskGraph(
            task_cost={
                SimpleTaskGraph.Cost.LOW: 1,
                SimpleTaskGraph.Cost.HIGH: 2,
            },
            data_cost={
                SimpleTaskGraph.Cost.LOW: 1,
                SimpleTaskGraph.Cost.HIGH: 2,
            }
        )

        nodes = []
        task_count = 0
        for i, level in enumerate(self.levels):
            nodes.append([])
            for _ in range(level):
                task_count += 1

                nodes[-1].append(task_count)
                task_graph.add_task(task_count, random.choice(choices))
                if i == 0:
                    continue
                for src in nodes[-2]:
                    task_graph.add_dependency(src, task_count, random.choice(choices))


        return task_graph
    
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
        loc="lower left"
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

    # _generator = CyclesGenerator(100)
    _generator = ExampleGenerator([5, 3, 1])
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

    saveGanttChart(proc_sched, savedir.joinpath("mother_network_schedule.png"))

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

    optimizer_exhaustive = ExhaustiveSearch(
        networks=mother_network.iter_subnetworks(),
        cost_func=cost_func
    )

    optimizer_simulated_annealing = SimulatedAnnealingOptimizer(
        start_network=mother_network.random_subnetwork(),
        get_neighbor=mother_network.random_neighbor,
        cost_func=cost_func,
        n_iterations=MAX_ITERATIONS,
        accept=ExponentialCool(
            initial_temperature=10,
            base=1/np.e
        )
    )

    columns=["optimizer", "iteration", "avg_makespan", "deploy_cost", "risk", "cost"]
    rows = []
    for iteration, (res_exhaustive, res_simulated_annealing) in enumerate(
            zip(optimizer_exhaustive.optimize_iter(), optimizer_simulated_annealing.optimize_iter()),
            start=1
        ):
        for optimizer, res in [("exhaustive", res_exhaustive), ("simulated_annealing", res_simulated_annealing)]:
            rows.append([
                optimizer,
                iteration,
                np.inf if not res.metadata["makespans"] else np.mean(res.metadata["makespans"]),
                res.metadata["deploy_cost"],
                res.metadata["risk"],
                res.cost,
            ])
            print(dict(zip(columns, rows[-1])))

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(savedir.joinpath("results.csv"),index=None)


if __name__ == "__main__":
    main()
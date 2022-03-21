from copy import deepcopy
import math
import random
from typing import List, Optional, Tuple
import matplotlib
from matplotlib import patches
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from nsdc.generators.generator import TaskGraphGenerator, TaskGraphSetGenerator
from nsdc.network.simple_network import SimpleNetwork
from nsdc.optimizers.optimizer import Result
from nsdc.optimizers.simulated_annealing import ExponentialCool, SimulatedAnnealingOptimizer
from nsdc.schedulers.heft import HeftScheduler
from nsdc.task_graph.simple_task_graph import SimpleTaskGraph
import numpy as np
import pandas as pd
from plane_network import PlaneNetworkFamily
import pathlib
from heft.gantt import showGanttChart
import warnings
warnings.filterwarnings('ignore')

thisdir = pathlib.Path(__file__).resolve().parent
savedir = thisdir.joinpath("outputs")
savedir.mkdir(parents=True, exist_ok=True)

NUM_TASK_GRAPHS = 5
COEF_MAKESPAN = 1/10
COEF_RISK = 1
COEF_DEPLOY_COST = 1/10

OUTLIER_COST = 100
MAX_ITERATIONS = 100


def saveGanttChart(proc_schedules, path: pathlib.Path):
    showGanttChart(proc_schedules)
    plt.savefig(path)
    plt.close()

def get_network_family() -> PlaneNetworkFamily:
    def satellite_bandwidth(pos: Tuple[float, float]) -> SimpleNetwork.Speed:
        x, y = pos 
        if x >= 0.5 and y >= 0.5:
            return SimpleNetwork.Speed.NONE
        elif y >= 0.5:
            return SimpleNetwork.Speed.LOW
        else:
            return SimpleNetwork.Speed.HIGH
        
    def radio_bandwidth(p1: Tuple[float, float], p2: Tuple[float, float]) -> SimpleNetwork.Speed:
        x1, y1 = p1 
        x2, y2 = p2 
        d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        if d < 0.1:
            return SimpleNetwork.Speed.HIGH
        elif d < 0.3:
            return SimpleNetwork.Speed.LOW
        else:
            return SimpleNetwork.Speed.NONE
    
    def gray_bandwidth(p1: Tuple[float, float], p2: Tuple[float, float]) -> SimpleNetwork.Speed:
        x1, y1 = p1 
        x2, y2 = p2 
        if x1 <= 0.3 and x2 <= 0.3:
            if y1 >= 0.5 or y2 >= 0.5:
                return SimpleNetwork.Speed.LOW
            else:
                return SimpleNetwork.Speed.HIGH
        else:
            return SimpleNetwork.Speed.NONE


    network_family = PlaneNetworkFamily(
        satellite_bandwidth=satellite_bandwidth,
        radio_bandwidth=radio_bandwidth,
        gray_bandwidth=gray_bandwidth,
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

    network_family.add_node(0, SimpleNetwork.Speed.LOW, pos=(0.1, 0.25))
    network_family.add_node(1, SimpleNetwork.Speed.LOW)
    network_family.add_node(2, SimpleNetwork.Speed.LOW)
    network_family.add_node(3, SimpleNetwork.Speed.HIGH)
    network_family.add_node(4, SimpleNetwork.Speed.HIGH, pos=(0.6, 0.6))

    return network_family

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
    
    
def draw_network(network: SimpleNetwork, 
                 path: Optional[pathlib.Path] = None,
                 subnetwork: Optional[SimpleNetwork] = None,
                 title: str = "Network",
                 ax: Optional[plt.Axes] = None) -> None:
    fig, ax = network.draw(subnetwork or network, ax=ax)
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
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if path is not None:
        fig.savefig(path)
    return fig, ax

def draw_task_graph(path: pathlib.Path, 
                    task_graph: SimpleTaskGraph,
                    title: str = "Task Graph",
                    ax: Optional[plt.Axes] = None) -> None:
    fig, ax = task_graph.draw(ax)
    ax.set_title(title)
    ax.legend(
        handles=[
            patches.Patch(color='#BF616A', label='High Cost/Data'),
            patches.Patch(color='#EBCB8B', label='Low Cost/Data')
        ],
        loc="lower left"
    )
    ax.axis('off')
    ax.margins(0)
    fig.savefig(path)

def main():
    network_family = get_network_family()
    
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
    
    optimizer = SimulatedAnnealingOptimizer(
        start_network=network_family.random_network(),
        get_neighbor=network_family.random_neighbor,
        cost_func=cost_func,
        n_iterations=MAX_ITERATIONS,
        accept=ExponentialCool(
            initial_temperature=10,
            base=1/np.e
        )
    )
    
    columns=["iteration", "avg_makespan", "deploy_cost", "risk", "cost"]
    rows = []
    best_cost: float = np.inf
    best_network: Optional[SimpleNetwork] = None 
    ZFILL = math.ceil(np.log10(MAX_ITERATIONS+1))
    for iteration, res in enumerate(optimizer.optimize_iter(), start=1):
        rows.append([
            iteration,
            np.inf if not res.metadata["makespans"] else np.mean(res.metadata["makespans"]),
            res.metadata["deploy_cost"],
            res.metadata["risk"],
            res.cost,
        ])
        print(dict(zip(columns, rows[-1])))

        if res.cost < best_cost:
            best_cost, best_network = res.cost, deepcopy(res.network)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        savedir.joinpath("networks").mkdir(exist_ok=True, parents=True)
        draw_network(
            network=res.network,
            title=f"Network (Iteration {str(iteration).zfill(ZFILL)})",
            ax=ax1
        )
        draw_network(
            network=best_network,
            title=f"Best Network (Iteration {str(iteration).zfill(ZFILL)})",
            ax=ax2
        )
        
        fig.savefig(
            savedir.joinpath("networks", f"iteration_{str(iteration).zfill(ZFILL)}.png")
        )
        
    draw_network(
        path=savedir.joinpath("best_network.png"),
        network=best_network,
        title="Best Network"
    )
    
    task_graph = task_graph_generator.generate()
    proc_sched, task_sched, dict_sched  = scheduler.get_schedule(
        network=best_network,
        task_graph=task_graph
    )
    saveGanttChart(proc_sched, savedir.joinpath("best_network_schedule.png"))

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(savedir.joinpath("results.csv"),index=None)

if __name__ == "__main__":
    main()
from bnet.optimizers.optimizer import Result
from matplotlib.lines import Line2D
from bnet.task_graph.eugenio_simple_task_graph_generator import EugenioSimpleTaskGraphGenerator
from bnet.task_graph import SimpleTaskGraph
from bnet.network import SimpleNetwork
from bnet.optimizers.simulated_annealing import ExponentialCool, SimulatedAnnealingOptimizer
from bnet.schedulers import HeftScheduler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from bnet.generators.generator import CycleGenerator

from pprint import pprint


def main():
    SAMPLES = 5
    
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

    task_graph_generator = EugenioSimpleTaskGraphGenerator(
        nodes=10, 
        edges=15,
        task_cost={
            SimpleTaskGraph.Cost.LOW: 300,
            SimpleTaskGraph.Cost.HIGH: 500,
        },
        data_cost={
            SimpleTaskGraph.Cost.LOW: 300,
            SimpleTaskGraph.Cost.HIGH: 500,
        }
    )
    
    def cost_func(network: SimpleNetwork) -> Result:
        task_graphs = []
        makespans = []
        for i in range(SAMPLES):
            try:
                task_graphs.append(task_graph_generator.generate())
                makespans.append(scheduler.schedule(network, task_graphs[-1]))
            except Exception as e:
                print(e) 

        makespan = np.inf if not makespans else sum(makespans) / len(makespans)
        deploy_cost = network.cost()
        risk = network.risk()
        return Result(
            network=network,
            cost=deploy_cost + 20 * risk + makespan / 10,
            metadata={
                "task_graphs": task_graphs,
                "makespans": makespans,
                "deploy_cost": deploy_cost,
                "risk": risk,
            }
        )
    
    
    scheduler = HeftScheduler()

    res = {}
    for t in range(10, 1000, 100):
        optimizer = SimulatedAnnealingOptimizer(
            mother_network=mother_network,
            start_network=mother_network.random_subnetwork(),
            get_neighbor=mother_network.random_neighbor,
            cost_func=cost_func,
            n_iterations=100,
            accept=ExponentialCool(
                initial_temperature=t,
                base=1/np.e
            )
        )

        res[t] = optimizer.optimize().cost
        print(t, res[t])

    pprint(res)

if __name__ == "__main__":
    main()
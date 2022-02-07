import subprocess
import numpy as np
from bnet.optimizers.simulated_annealing import SimulatedAnnealingOptimizer
from bnet.optimizers.brute_force import BruteForceOptimizer
from bnet.optimizers.optimizer import Result
from bnet.task_graph.eugenio_simple_task_graph_generator import EugenioSimpleTaskGraphGenerator
from bnet.task_graph import SimpleTaskGraph
from bnet.network import SimpleNetwork
from bnet.schedulers import HeftScheduler
from bnet.generators.generator import TaskGraphSetGenerator

import argparse
import dill as pickle 
import pathlib 
from atomicwrites import atomic_write

thisdir = pathlib.Path(__file__).resolve().parent

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

    task_graph_generator = TaskGraphSetGenerator(
        task_graphs=[
            _task_graph_generator.generate()
            for _ in range(args.samples)
        ]
    )

    scheduler = HeftScheduler()
    def cost_func(network: SimpleNetwork) -> Result:
        task_graphs = []
        makespans = []
        for i in range(args.samples):
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

    bf_optimizer = BruteForceOptimizer(
        networks=mother_network.iter_subnetworks(),
        cost_func=cost_func
    )

    sa_optimizer = SimulatedAnnealingOptimizer(
        mother_network=mother_network,
        start_network=mother_network.random_subnetwork(),
        get_neighbor=mother_network.random_neighbor,
        cost_func=cost_func,
        n_iterations=100,
        initial_temperature=10
    )

    # # Start GUI 
    proc = subprocess.Popen(["python", str(thisdir.joinpath("app.py"))])

    # Start Optimization    
    thisdir.joinpath("mother.pickle").write_bytes(pickle.dumps(mother_network))
    bf_results, sa_results = [], []
    for i, (bf_result, sa_result) in enumerate(zip(bf_optimizer.optimize_iter(), sa_optimizer.optimize_iter())):
        bf_result.metadata["seq"] = i
        sa_result.metadata["seq"] = i
        bf_results.append(bf_result)
        sa_results.append(sa_result)
        with atomic_write(str(thisdir.joinpath("bf_result.pickle")), mode='wb', overwrite=True) as fp:
            fp.write(pickle.dumps(bf_results))
        with atomic_write(str(thisdir.joinpath("sa_result.pickle")), mode='wb', overwrite=True) as fp:
            fp.write(pickle.dumps(sa_results))
        # print(f"{bf_result.metadata['seq']}: BF={bf_result.cost}, SA={sa_result.cost}")

    
    # # Wait so GUI doesn't close
    proc.wait()

if __name__ == "__main__":
    main()
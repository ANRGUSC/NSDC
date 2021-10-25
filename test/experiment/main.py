import random
import numpy as np
from bnet.optimizers.simulated_annealing import ExponentialCool, SimulatedAnnealingOptimizer
from bnet.optimizers.brute_force import BruteForceOptimizer
from bnet.optimizers.optimizer import Result
from bnet.task_graph.eugenio_simple_task_graph_generator import EugenioSimpleTaskGraphGenerator
from bnet.task_graph import SimpleTaskGraph
from bnet.network import SimpleNetwork
from bnet.schedulers import HeftScheduler
from bnet.generators.generator import CycleGenerator

import dill as pickle 
from itertools import product
import wandb
import codecs

from typing import Any, Dict, Generator, Union, List, Tuple, Set

def get_combos(config: Dict[str, Union[List, Tuple, Set, Dict]]):
    vars = {
        key: list(
            value if isinstance(value, (tuple, list, set, dict)) else (
                value.keys() if isinstance(value, dict) else 
                [value]
            )
        )
        for key, value in config.items()
    }

    subconfigs = {key for key, value in config.items() if isinstance(value, dict)}
    for values in product(*vars.values()):
        combo = dict(zip(vars.keys(), values))
        if not subconfigs:
            yield combo 
        else:
            for subconfig in subconfigs:
                combo.update(config[subconfig][combo[subconfig]])
            yield from get_combos(combo)
        

def get_configs() -> Generator[Dict[str, Any], None, None]:
    VARIABLES = dict(
        runs_per_configuration = 4,
        task_graph_samples = 5,
        task_graph_cycle = True,

        node_speed_low = 10,
        node_speed_high = 100,
        sat_speed_low = 5,
        sat_speed_high = 10,
        radio_speed_low = 2,
        radio_speed_high = 4,
        gray_speed_low = 100,
        gray_speed_high = 200,

        network_order = [10, 50, 100],
        task_graph_order = [10, 50, 100],

        task_cost_low = 300,
        task_cost_high = 500,
        data_cost_low = 300,
        data_cost_high = 500,

        optimizer={
            "simulated_annealing": dict(
                sa_neighborhood=["random_connected_neighbor", "random_neighbor"],
                sa_initial_temperature = [10, 50, 100],
                sa_base = [0, np.e, 1],
            )
        },
        
        max_iterations = 100,
        scheduler = "heft",
        task_graph_generator = "eugenio", # "simple"
    )
    yield from get_combos(VARIABLES)

def main():
    configs = list(get_configs())
    for config_id, config in enumerate(configs):
        print(f"Config {config_id}/{len(configs)} ({config_id/len(configs)*100:.2f}%)")
        for run_id in range(config["runs_per_configuration"]):
            print(f"  Run {run_id}")
            if config["task_graph_generator"] == "simple":
                task_graph_generator = SimpleTaskGraph.random_generator(
                    num_tasks=config["task_graph_order"],
                    data_cost={
                        SimpleTaskGraph.Cost.LOW: config["data_cost_low"],
                        SimpleTaskGraph.Cost.HIGH: config["data_cost_high"],
                    }
                ) 
            elif config["task_graph_generator"] == "eugenio":
                task_graph_generator = EugenioSimpleTaskGraphGenerator(
                    nodes=config["task_graph_order"], 
                    edges=random.randint(config["task_graph_order"]-1, config["task_graph_order"] * (config["task_graph_order"] - 1) // 2),
                    task_cost={
                        SimpleTaskGraph.Cost.LOW: config["task_cost_low"],
                        SimpleTaskGraph.Cost.HIGH: config["task_cost_high"],
                    },
                    data_cost={
                        SimpleTaskGraph.Cost.LOW: config["data_cost_low"],
                        SimpleTaskGraph.Cost.HIGH: config["data_cost_high"],
                    }
                )
            else:
                raise ValueError(f'Invalid task graph generator: {config["task_graph_generator"]}')

            if config["task_graph_cycle"]:
                task_graph_generator = CycleGenerator(
                    task_graphs=[
                        task_graph_generator.generate() 
                        for _ in range(config["task_graph_samples"])
                    ]
                )
            
            if config["scheduler"] == "heft":
                scheduler = HeftScheduler()
            else:
                raise ValueError(f'Invalid scheduler: {config["scheduler"]}')

            mother_network = SimpleNetwork.random(
                config["network_order"],
                node_speed={
                    SimpleNetwork.Speed.LOW: config["node_speed_low"],        # Node speed, a node with speed s computes a task with size t in n/s seconds
                    SimpleNetwork.Speed.HIGH: config["node_speed_high"],
                },
                sat_speed={
                    SimpleNetwork.Speed.LOW: config["sat_speed_low"],         # Satellite speed, an edge with speed (trasnfer rate) s transmits data of size d between two nodes in s/d seconds 
                    SimpleNetwork.Speed.HIGH: config["sat_speed_high"],
                },
                radio_speed={
                    SimpleNetwork.Speed.LOW: config["radio_speed_low"],
                    SimpleNetwork.Speed.HIGH: config["radio_speed_high"],
                },
                gray_speed={
                    SimpleNetwork.Speed.LOW: config["gray_speed_low"],
                    SimpleNetwork.Speed.HIGH: config["gray_speed_high"],
                }
            )

            def cost_func(network: SimpleNetwork) -> Result:
                task_graphs = []
                makespans = []
                for i in range(config["task_graph_samples"]):
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
        
            if config["optimizer"] == "brute_force":
                optimizer = BruteForceOptimizer(
                    networks=mother_network.iter_subnetworks(),
                    cost_func=cost_func
                )
            elif config["optimizer"] == "simulated_annealing":
                optimizer = SimulatedAnnealingOptimizer(
                    mother_network=mother_network,
                    start_network=mother_network.random_subnetwork(),
                    get_neighbor=(
                        mother_network.random_neighbor 
                        if config["sa_neighborhood"] == "random_neighbor" 
                        else mother_network.random_connected_neighbor
                    ),
                    cost_func=cost_func,
                    n_iterations=config["max_iterations"],
                    accept=ExponentialCool(
                        initial_temperature=config["sa_initial_temperature"],
                        base=config["sa_base"]
                    )
                )

            with wandb.init(reinit=True, project="iobt_ns", entity="anrg-iobt_ns", config=config) as run:
                result: Result
                best_cost: float = np.inf 
                for i, result in zip(range(config["max_iterations"]), optimizer.optimize_iter()):
                    if result.cost < best_cost:
                        best_cost = result.cost 
                    run.log({
                        "iteration": i, 
                        "cost": result.cost, 
                        "best_cost": best_cost, 
                        "mother_network": codecs.encode(pickle.dumps(mother_network), "base64").decode(),
                        "network": codecs.encode(pickle.dumps(result.network), "base64").decode(),
                        "metadata": codecs.encode(pickle.dumps(result.metadata), "base64").decode(),
                    }) 
            return 


if __name__ == "__main__":
    main()
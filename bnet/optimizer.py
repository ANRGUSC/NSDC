import networkx as nx 
from typing import Callable, Iterable, List, Union

from heft.heft import schedule_dag
import numpy as np

def find_network(networks: Iterable[nx.Graph], 
                 task_graph_sampler: Callable[[], nx.DiGraph],
                 scheduler: Callable[[nx.Graph, nx.DiGraph], float],
                 samples: Union[int, Callable[[List[float]], bool]]) -> nx.Graph:
    if isinstance(samples, (int, float)):
        do_sample = lambda metrics: len(metrics) < samples 
    else:
        do_sample = samples 

    min_makespan, opt_network = None, None 
    for network in networks:
        print(f"testing network: {list(network.nodes)}")
        makespans = []
        while do_sample(makespans):
            dag = task_graph_sampler()
            comp = np.array([
                [
                    network.nodes[node]["comp_multiplier"] * dag.nodes[task]["comp"]
                    for task in dag.nodes
                ]
                for node in network.nodes
            ])
            comm = nx.to_numpy_array(network)

            proc_schedule, task_schedule, res = schedule_dag(
                dag, 
                computation_matrix=comp, 
                communication_matrix=comm,
                communication_startup=np.zeros(network.order())
            )

            makespans.append(max([task.end for _, task in task_schedule.items()]))
        
        avg_makespan = sum(makespans) / len(makespans)
        if min_makespan is None or avg_makespan < min_makespan:
            min_makespan, opt_network = avg_makespan, network
    
    return min_makespan, opt_network

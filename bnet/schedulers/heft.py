from bnet import Network, TaskGraph
from heft.heft import schedule_dag
from .scheduler import Scheduler
import networkx as nx 
import numpy as np

class HeftScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()

    def schedule(self, task_graph: TaskGraph, network: Network) -> float:
        dag = task_graph.to_networkx()
        comp = task_graph.computation_matrix(network)
        comm = network.communication_matrix()

        network_relabel = {node: i for i, node in enumerate(comm.columns)}
        task_relabel = {task: i for i, task in enumerate(comp.index.values)}
        nx.set_edge_attributes(
            dag, 
            nx.get_edge_attributes(dag, "data"), 
            name="weight"
        )

        proc_sched, task_sched, dict_sched = schedule_dag(
            nx.relabel_nodes(dag, task_relabel),
            communication_matrix=comm.rename(
                index=network_relabel, 
                columns=network_relabel
            ).values,
            computation_matrix=comp.rename(
                index=task_relabel, 
                columns=network_relabel
            ).values,
            communication_startup=np.zeros(len(network.nodes))
        )

        return max([event.end for _, event in task_sched.items()])
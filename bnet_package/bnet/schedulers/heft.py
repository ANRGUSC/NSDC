from typing import Callable, Dict, Hashable, Iterable, Optional
from bnet import Network, TaskGraph
from heft.heft import schedule_dag
from .scheduler import Scheduler
import networkx as nx 
import numpy as np
import pandas as pd 

class HeftScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()

    def get_schedule(self, 
                     network: Network, 
                     task_graph: TaskGraph,
                     can_execute: Callable[[Hashable, Hashable], bool] = lambda *_: True):
        dag = task_graph.to_networkx()
        comp = task_graph.computation_matrix(network, can_execute)
        comm = network.bandwidth_matrix()

        # Add dummy source and destination nodes to task graph 
        comp = comp.append(
            pd.DataFrame(
                np.zeros((2, len(comp.columns))), 
                index=["SRC", "DST"],
                columns=comp.columns
            )
        )

        dag.add_node("SRC", cost=0)
        dag.add_node("DST", cost=0)
        for node in list(dag.nodes):
            if node in ["SRC", "DST"]:
                continue
            if dag.in_degree(node) == 0:
                dag.add_edge("SRC", node, data=0)
            if dag.out_degree(node) == 0:
                dag.add_edge(node, "DST", data=0)

        network_relabel = {node: i for i, node in enumerate(comm.columns)}
        task_relabel = {task: i for i, task in enumerate(comp.index.values)}
        nx.set_edge_attributes(
            dag, 
            nx.get_edge_attributes(dag, "data"), 
            name="weight"
        )

        comm_matrix = 1 / comm.rename(
            index=network_relabel,
            columns=network_relabel
        ).values
        comp_matrix = comp.rename(
            index=task_relabel, 
            columns=network_relabel
        ).values

        comm_matrix[comm_matrix == np.inf] = 1e-7
        np.fill_diagonal(comm_matrix, 0)

        # comm_matrix = np.nan_to_num(comm_matrix)
        # print(comm_matrix)
        # print(comp_matrix)

        return schedule_dag(
            nx.relabel_nodes(dag, task_relabel),
            communication_matrix=comm_matrix,
            computation_matrix=comp_matrix,
            communication_startup=np.zeros(len(network.nodes))
        )

    def schedule(self, 
                 network: Network, 
                 task_graph: TaskGraph,
                 can_execute: Callable[[Hashable, Hashable], bool] = lambda *_: True) -> float:
        proc_sched, task_sched, dict_sched = self.get_schedule(network, task_graph, can_execute)
        return max([event.end for _, event in task_sched.items()])
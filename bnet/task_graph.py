from abc import ABC
from typing import Callable, Dict, Tuple
import networkx as nx
import pandas as pd 
from .network import Network


class TaskGraph(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.tasks: Dict[str, Callable[[float], float]] = {}
        self.edges: Dict[Tuple[str, str], float] = {}
        self.nx_graph = nx.DiGraph()

    def add_task(self, 
                 name: str, 
                 exec_time: Callable[[float], float]) -> None:
        self.tasks[name] = exec_time
    
    def add_dependency(self, 
                       src: str, 
                       dst: str, 
                       data: float) -> None:
        if not src in self.tasks:
            raise ValueError(f"Task {src} has not beed added")
            
        if not dst in self.tasks:
            raise ValueError(f"Task {dst} has not beed added")
        self.nx_graph.add_edge(src, dst, data=float(data))
        if not nx.is_directed_acyclic_graph(self.nx_graph):
            self.nx_graph.remove_edge(src, dst)
            raise ValueError(f"Adding {(src, dst)} would create a cycle")
        self.edges[(src, dst)] = float(data) 

    def computation_matrix(self, network: Network) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [
                    exec_time(speed)
                    for _, speed in network.nodes.items()
                ]
                for _, exec_time in self.tasks.items()
            ],
            columns=network.nodes.keys(),
            index=self.tasks.keys()
        )
        
    def to_networkx(self) -> nx.DiGraph:
        return self.nx_graph.copy()

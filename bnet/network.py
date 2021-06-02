from abc import ABC
from typing import Dict, Tuple
import networkx as nx
import numpy as np
import pandas as pd 

class Network(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.nodes: Dict[str, float] = {}
        self.edges: Dict[Tuple[str, str], float] = {}

    def add_node(self, 
                 name: str, 
                 speed: float) -> None:
        self.nodes[name] = float(speed)
        self.add_edge(name, name, np.inf)

    def add_edge(self, src: str, dst: str, bandwidth: float) -> None:
        self.edges[(src, dst)] = float(bandwidth)
        if (dst, src) in self.edges:
            del self.edges[(dst, src)]

    def bandwidth_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [
                    self.edges.get((n1, n2), self.edges.get((n2, n1), np.inf))
                    for n2 in self.nodes.keys()
                ]
                for n1 in self.nodes.keys()
            ],
            columns=self.nodes.keys(),
            index=self.nodes.keys()
        )

    def communication_matrix(self) -> pd.DataFrame:
        return 1 / self.bandwidth_matrix()

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_edges_from([
            (src, dst, {"bandwidth": bandwidth}) 
            for (src, dst), bandwidth in self.edges.items()
        ])
        nx.set_node_attributes(graph, self.nodes, name="speed")
        return graph 

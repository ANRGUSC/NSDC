from abc import ABC, abstractmethod
from typing import List, Tuple
import networkx as nx
import pandas as pd 

class NoEdgeError(Exception):
    pass 

class Network(ABC):
    def __init__(self) -> None:
        super().__init__()

        
    @property
    @abstractmethod
    def nodes(self) -> List[str]:
        """List of nodes in the network"""
        raise NotImplementedError

    @property
    @abstractmethod
    def edges(self) -> List[Tuple[str, str, str]]:
        """List of edges in the network"""
        raise NotImplementedError

    @abstractmethod
    def add_node(self, name: str, speed: float) -> None:
        """Add node with speed to the network

        Args:
            name: Name of node.
            speed: Speed of node, used to compute the time to execute a task on the node. \
                A task with cost c is estimated to take c*speed time to execute on the node.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_node(self, name: str) -> None:
        """Removes a node from the graph

        Args:
            name: Name of node to remove.
        """
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, src: str, dst: str, bandwidth: float) -> None:
        """Adds an edge to the network

        Args:
            src: source of edge
            dst: destination of edge
            bandwidth: bandwith between src and dst
        """
        raise NotImplementedError

    @abstractmethod
    def remove_edge(self, src: str, dst: str) -> None:
        """Removes an edge from the graph

        Args:
            src: Source of edge to remove.
            dst: Destination of edge to remove.
        """
        raise NotImplementedError

    @abstractmethod
    def bandwidth_matrix(self) -> pd.DataFrame:
        """Returns a pandas DataFrame of the node-to-node bandwidth

        Returns:
            pd.DataFrame: Rows and columns are nodes in the network. Cells are \
                bandwidth between nodes in the network.
        """
        raise NotImplementedError

    def communication_matrix(self) -> pd.DataFrame:
        """Returns a pandas DataFrame of the node-to-node communication cost

        Communication cost is 1 / bandwidth

        Returns:
            pd.DataFrame: Rows and columns are nodes in the network. Cells are \
                communication cost between nodes in the network.
        """
        return 1 / self.bandwidth_matrix()

    @abstractmethod
    def to_networkx(self) -> nx.Graph:
        """Converts the network into a networkx representation.

        The networkx graph should have "speed" node attribute and "bandwidth" edge attribute.

        Returns:
            nx.Graph: NetworkX representation of network.
        """
        raise NotImplementedError

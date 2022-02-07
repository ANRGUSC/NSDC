from abc import ABC, abstractmethod
from typing import Callable, Dict, Hashable, Iterable, Optional

import networkx as nx
import pandas as pd 
from ..network import Network


class TaskGraph(ABC):
    def __init__(self) -> None:
        """Constructor for TaskGraph"""
        super().__init__()

    @abstractmethod
    def add_task(self, name: str, cost: Callable[[float], float]) -> None:
        """Adds a task to the task graph

        Args:
            name: Name for task.
            cost: Cost of task.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_task(self, name: str) -> None:
        """Removes a task from the task graph

        Args:
            name: Name for task.
        """
        raise NotImplementedError

    
    @abstractmethod
    def add_dependency(self, src: str, dst: str, data: float) -> None:
        """Adds a dependency to the task graph

        Args:
            src: Source task which sends data to dst.
            dst: Destination task which depends on the output of src.
            data: Amount of data to be sent between tasks. This value interacts with the \
                bandwidth of the edge the data is sent over. The data takes data / b \
                to be sent over a channel with b bandwidth.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_dependency(self, src: str, dst: str) -> None:
        """Removes a dependency from the task graph

        Args:
            src: Source task which sends data to dst.
            dst: Destination task which depends on the output of src.
        """
        raise NotImplementedError

    @abstractmethod
    def computation_matrix(self, 
                           network: Network,
                           can_execute: Callable[[Hashable, Hashable], bool] = lambda *_: True) -> pd.DataFrame:
        """Returns computation matrix for a given task graph and network
        
        Returns DataFrame that estimates how long each task would take to execute on \
        each node in a network

        Args:
            network: Network that task graph will execute on
        Returns:
            pd.DataFrame: Columns are nodes in the network and rows are tasks. Cells \
                contain the estimated amount of time to execute the task on a given node \
                in the network. 
        """
        raise NotImplementedError
        
    def to_networkx(self) -> nx.DiGraph:
        """Converts the task graph into a networkx directed graph representation.

        The networkx directed graph should have "cost" node attribute and "data" edge attribute.

        Returns:
            nx.DiGraph: NetworkX directed graph representation of network.
        """
        raise NotImplementedError

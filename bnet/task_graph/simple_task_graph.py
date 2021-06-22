import numpy as np
from bnet.network import Network
import random

from numpy.random import zipf
from enum import Enum
from typing import Callable, Dict, Optional, Union
import networkx as nx
import pandas as pd 
from .task_graph import TaskGraph
from bnet.generators.generator import TaskGraphGenerator

class SimpleTaskGraph(TaskGraph):
    """Simple task graph with limited cost factors NONE, LOW, and HIGH"""
    class Cost(Enum):
        NONE = 0
        LOW = 1
        HIGH = 2
    
    CostConverter = Optional[
        Union[
            Callable[["SimpleTaskGraph.Cost"], float], 
            Dict["SimpleTaskGraph.Cost", float]
        ]
    ]
    
    def __init__(self,
                 task_cost: "SimpleTaskGraph.CostConverter" = None,
                 data_cost: "SimpleTaskGraph.CostConverter" = None) -> None:
        """Constructor for SimpleTaskGraph
        
        Args:
            task_cost: Function that converts a SimpleTaskGraph.Cost (NONE, LOW, HIGH) to an actual \
                task cost (float). By Default, NONE=0, LOW=1, and HIGH=2.
            data_cost: Function that converts a SimpleTaskGraph.Cost (NONE, LOW, HIGH) to an actual \
                edge data cost (float) for task dependency edges. By Default, NONE=0, LOW=1, and HIGH=2.
        """
        super().__init__()
        def cost_func(converter: "SimpleTaskGraph.CostConverter") -> Callable[["SimpleTaskGraph.Cost"], float]:
            if converter is None:
                func = lambda x: x.value
            elif isinstance(converter, dict):
                func = lambda x: converter[x]
            else:
                func = converter
            def _func(speed: SimpleTaskGraph.Cost) -> float:
                if speed == SimpleTaskGraph.Cost.NONE:
                    return 0
                return func(speed)
            return _func 
        self._graph = nx.DiGraph()
        self.task_cost = cost_func(task_cost)
        self.data_cost = cost_func(data_cost)

    def add_task(self, name: str, cost: "SimpleTaskGraph.Cost") -> None:
        """Adds a task to the task graph
        
        Args:
            name: name of task to add
            cost: cost of task to add (NONE, LOW, HIGH)
        """
        if cost != SimpleTaskGraph.Cost.NONE:
            self._graph.add_node(name, cost=cost)

    def add_dependency(self, src: str, dst: str, data: "SimpleTaskGraph.Cost") -> None:
        """Adds a dependency to the task graph
        
        Args:
            src: Source task which sends data to dst.
            dst: Destination task which depends on the output of src.
            data: Amount of data to be sent between tasks. This value interacts with the \
                bandwidth of the edge the data is sent over. The data takes data / b \
                to be sent over a channel with b bandwidth.
        """
        if data != SimpleTaskGraph.Cost.NONE:
            self._graph.add_edge(src, dst, data=data)

    def computation_matrix(self, network: Network) -> pd.DataFrame:   
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
        network_graph = network.to_networkx()
        return pd.DataFrame(
            [
                [
                    np.inf if network_graph.nodes[node]["speed"] == 0 else self.task_cost(self._graph.nodes[task]["cost"]) / network_graph.nodes[node]["speed"]
                    for node in network_graph.nodes
                ]
                for task in self._graph.nodes
            ],
            columns=list(network_graph.nodes),
            index=list(self._graph.nodes)
        )

    def to_networkx(self) -> nx.DiGraph:
        """Converts the task graph into a networkx directed graph representation.

        The networkx directed graph should have "cost" node attribute and "data" edge attribute.

        Returns:
            nx.DiGraph: NetworkX directed graph representation of network.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(self._graph.nodes)
        nx.set_node_attributes(
            graph, 
            {
                node: self.task_cost(self._graph.nodes[node]["cost"]) 
                for node in self._graph.nodes
            }, 
            name="cost"
        )

        data = {
            (src, dst): self.data_cost(self._graph.edges[(src, dst)]["data"])
            for src, dst in self._graph.edges
        }
        graph.add_edges_from(data.keys())
        nx.set_edge_attributes(graph, data, name="data")

        return graph 

    @classmethod
    def random_generator(cls, 
                         num_tasks: int,
                         task_cost: "SimpleTaskGraph.CostConverter" = None,
                         data_cost: "SimpleTaskGraph.CostConverter" = None) -> "SimpleTaskGraphGenerator":
        """Generates a random SimpleTaskGraph

        Generates a random simple task graph by first generating a complete graph and adding
        edges to the task graph as long as doing so would not create a cycle.
        
        Args:
            num_nodes: Number of nodes in graph. 
            task_cost: Function that converts a SimpleTaskGraph.Cost (NONE, LOW, HIGH) to an actual \
                task cost (float). By Default, NONE=0, LOW=1, and HIGH=2.
            data_cost: Function that converts a SimpleTaskGraph.Cost (NONE, LOW, HIGH) to an actual \
                edge data cost (float) for task dependency edges. By Default, NONE=0, LOW=1, and HIGH=2.
        """
        return SimpleTaskGraphGenerator(lambda: num_tasks, task_cost, data_cost)

    @classmethod
    def random_generator_zipf(cls, 
                              zipf_constant: float,
                              zipf_low: int = 10,
                              task_cost: "SimpleTaskGraph.CostConverter" = None,
                              data_cost: "SimpleTaskGraph.CostConverter" = None) -> "SimpleTaskGraphGenerator":
        """Generates a random SimpleTaskGraph

        Generates a random simple task graph by first generating a complete graph and adding
        edges to the task graph as long as doing so would not create a cycle.
        Uses a Zipf distribution to determine the number of tasks to create
        
        Args:
            zipf_constant: Zipf constant for generating task graphs with different sizes. 
            zipf_low: Minimum number of tasks to generate. Default is 10.
            task_cost: Function that converts a SimpleTaskGraph.Cost (NONE, LOW, HIGH) to an actual \
                task cost (float). By Default, NONE=0, LOW=1, and HIGH=2.
            data_cost: Function that converts a SimpleTaskGraph.Cost (NONE, LOW, HIGH) to an actual \
                edge data cost (float) for task dependency edges. By Default, NONE=0, LOW=1, and HIGH=2.
        """
        return SimpleTaskGraphGenerator(lambda: zipf(zipf_constant) + zipf_low, task_cost, data_cost)

class SimpleTaskGraphGenerator(TaskGraphGenerator):
    """Simple Task Graph Generator
    
    Generates a random simple task graph by first generating a complete graph and adding
    edges to the task graph as long as doing so would not create a cycle.
    """
    def __init__(self, 
                 random_num_tasks: Callable[[], int],
                 task_cost: "SimpleTaskGraph.CostConverter" = None,
                 data_cost: "SimpleTaskGraph.CostConverter" = None) -> None:
        """Constructor for Simple Task Graph Generator

        Args:
            num_tasks: Callable to get random number of tasks to generate.
        """
        super().__init__()
        self.task_cost = task_cost
        self.data_cost = data_cost
        self.random_num_tasks = random_num_tasks

    def random_cost(self, include_none: bool = False) -> SimpleTaskGraph.Cost:
        """Gets a random SimpleTaskGraph.Cost

        Args:
            include_none: If True, include SimpleTaskGraph.Cost.NONE in random chocies. \
                Default is False
        """
        choices = [SimpleTaskGraph.Cost.LOW, SimpleTaskGraph.Cost.HIGH]
        if include_none:
            choices.append(SimpleTaskGraph.Cost.NONE)
        return random.choice(choices) 

    def generate(self) -> SimpleTaskGraph:
        """Generates a random SimpleTaskGraph

        Generates a random simple task graph by first generating a complete graph and adding
        edges to the task graph as long as doing so would not create a cycle.
        """
        task_graph = SimpleTaskGraph(self.task_cost, self.data_cost)
        graph = nx.complete_graph(self.random_num_tasks())
        
        for task in graph.nodes():
            task_graph.add_task(task, cost=self.random_cost())

        for u, v in graph.edges():
            if u < v:
                task_graph.add_dependency(u, v, data=self.random_cost(include_none=False))

        return task_graph
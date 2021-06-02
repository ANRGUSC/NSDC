from bnet import Network, TaskGraph
from typing import Tuple
import networkx as nx 
import random 
from .generator import NetworkGenerator, TaskGraphGenerator

def rand_float(low: float, high: float) -> float:
    return random.random() * (high - low) + low 

class NaiveNetworkGenerator(NetworkGenerator):
    def __init__(self,
                 num_nodes: int, 
                 comp_multiplier_range: Tuple[float, float], 
                 comm_range: Tuple[float, float]) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.comp_multiplier_range = comp_multiplier_range
        self.comm_range = comm_range

    def generate(self) -> Network:
        network = Network()
        graph: nx.Graph = nx.complete_graph(self.num_nodes)

        for node in graph.nodes:
            network.add_node(node, speed=rand_float(*self.comp_multiplier_range))
        
        for u, v in graph.edges():
            network.add_edge(u, v, bandwidth=rand_float(*self.comp_multiplier_range))

        return network


class NaiveTaskGraphGenerator(TaskGraphGenerator):
    def __init__(self,
                 num_tasks: int, 
                 comp_range: Tuple[float, float], 
                 data_range: Tuple[float, float]) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.comp_range = comp_range
        self.data_range = data_range

    def generate(self) -> TaskGraph:
        task_graph = TaskGraph()
        graph = nx.complete_graph(self.num_tasks)
        
        for task in graph.nodes():
            multiplier = rand_float(*self.comp_range)
            task_graph.add_task(task, exec_time=lambda speed: speed * multiplier)

        for u, v in graph.edges():
             if u < v:
                 task_graph.add_dependency(u, v, data=rand_float(*self.data_range))

        return task_graph


class SimpleDAGGenerator(TaskGraphGenerator):
    def __init__(self, order: int, size: int) -> None:
        super().__init__()
        self.order = order 
        self.size = size 

    def generate(self) -> TaskGraph:
        # TODO: Create TaskGraph rather than networkx DiGraph
        G = nx.DiGraph()
        for i in range(self.order):
            G.add_node(i)
        while self.size > 0:
            a = random.randint(0, self.order - 1)
            b = a
            while b == a:
                b = random.randint(0, self.order - 1)
            G.add_edge(a, b)
            if nx.is_directed_acyclic_graph(G):
                self.size -= 1
            else: # we closed a loop!
                G.remove_edge(a,b)

        dag_dict = {}
        for node in G:
            dag_dict[node] = [n for n in G.neighbors(node)]
        
        dag = dag_dict
        return G, dag
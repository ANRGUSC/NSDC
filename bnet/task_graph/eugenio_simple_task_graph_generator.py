
from typing import Dict
from bnet.task_graph.simple_task_graph import SimpleTaskGraph
from bnet.task_graph.task_graph import TaskGraph
from bnet.generators.generator import TaskGraphGenerator

import random 
import networkx as nx 

class EugenioSimpleTaskGraphGenerator(TaskGraphGenerator):
    def __init__(self, 
                 nodes: int, 
                 edges: int,
                 task_cost: Dict[SimpleTaskGraph.Cost, float] = {}, 
                 data_cost: Dict[SimpleTaskGraph.Cost, float] = {}) -> None:
        super().__init__()
        self.nodes = nodes
        self.edges = edges 
        self.task_cost = task_cost
        self.data_cost = data_cost

    def generate(self) -> SimpleTaskGraph:
        task_graph = SimpleTaskGraph(self.task_cost, self.data_cost)
        for i in range(self.nodes):
            task_graph.add_task(
                i, cost=random.choice([
                    SimpleTaskGraph.Cost.LOW, 
                    SimpleTaskGraph.Cost.HIGH
                ])
            )
        
        edge_count = 0
        while edge_count < self.edges:
            a = random.randint(0, self.nodes - 1)
            b = a
            while b==a:
                b = random.randint(0, self.nodes - 1)
            task_graph.add_dependency(a, b, data=random.choice([
                    SimpleTaskGraph.Cost.LOW, 
                    SimpleTaskGraph.Cost.HIGH
                ])
            )
            if nx.is_directed_acyclic_graph(task_graph._graph):
                edge_count += 1
            else:
                # we closed a loop!
                task_graph.remove_dependency(a, b)
        
        return task_graph
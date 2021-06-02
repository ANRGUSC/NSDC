from bnet.generators.generator import TaskGraphGenerator
import numpy as np
from bnet import Network, TaskGraph
from bnet.schedulers import Scheduler
from typing import Callable, Generator, Iterable, Tuple
from .optimizer import Optimizer

from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx 
import matplotlib.pyplot as plt

class BruteForceOptimizer(Optimizer):
    def __init__(self, samples: int) -> None:
        super().__init__()
        self.samples = samples 

    def optimize_iter(self, 
                      networks: Iterable[Network], 
                      task_graph_generator: TaskGraphGenerator, 
                      scheduler: Scheduler) -> Generator[Tuple[Network, float, TaskGraph, Network, float], None, None]:
        min_score, best_network = np.inf, None
        for network in networks:
            cum_score = 0
            for _ in range(self.samples):
                task_graph = task_graph_generator.generate()
                score = scheduler.schedule(task_graph, network)

                cum_score += score
                yield network, score, task_graph, best_network, min_score
            
            avg_score = cum_score / self.samples
            if avg_score < min_score:
                min_score, best_network = avg_score, network 

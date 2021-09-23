from bnet.generators.generator import TaskGraphGenerator
import numpy as np
from bnet import Network, TaskGraph
from typing import Callable, Generator, Iterable
from .optimizer import Optimizer, Result

class BruteForceOptimizer(Optimizer):
    def __init__(self, 
                 networks: Iterable[Network], 
                 samples: int,
                 task_graph_generator: TaskGraphGenerator,
                 cost_func: Callable[..., float],
                 **metrics: Callable[[Network, TaskGraph], float]) -> None:
        super().__init__()
        self.networks = networks
        self.samples = samples 
        self.task_graph_generator = task_graph_generator
        self.cost_func = cost_func
        self.metrics = metrics

    def optimize_iter(self) -> Generator[Result, None, None]:
        min_cost, best_network, best_metrics = np.inf, None, {}
        for network in self.networks:
            cum_cost = 0
            cum_metrics = []
            for _ in range(self.samples):
                task_graph = self.task_graph_generator.generate()
                metrics = {
                    metric: func(network, task_graph) 
                    for metric, func in self.metrics.items()
                }
                cost = self.cost_func(**metrics)
                metrics["cost"] = cost
                cum_metrics.append(metrics)
                cum_cost += cost
                yield Result(
                    network, 
                    metrics, 
                    task_graph, 
                    best_network, 
                    best_metrics
                )
            
            avg_cost = cum_cost / self.samples
            if avg_cost < min_cost:
                min_cost, best_network, best_metrics = avg_cost, network, cum_metrics

        yield Result(
            network, 
            metrics, 
            task_graph, 
            best_network, 
            best_metrics
        )

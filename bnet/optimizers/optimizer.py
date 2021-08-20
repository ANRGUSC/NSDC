from abc import ABC, abstractmethod
from bnet.task_graph import TaskGraph
from typing import Any,  Tuple, Generator
from bnet import Network

class Optimizer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def optimize_iter(self) -> Generator[Tuple[Network, Any, TaskGraph, Network, Any], None, None]:
        """Optimize iterator which yields itermediate results 
    
        Yields:
            Tuple: tuple of five elements: (current network being considered, \
                metric(s) for current and/or current task graph, current task \
                graph being considered, best network found so far, metric(s) \
                for best network found so far)
        """
        pass

    def optimize(self) -> Tuple[Network, Any]:
        """Runs optimization and returns best result/metrics
        
        Returns:
            Tuple: tuple of two elements: (best network found so far, \
                metric(s) for best network found so far)
        """
        for *_, best_network, best_score in self.optimize_iter():
            pass 
        return best_network, best_score
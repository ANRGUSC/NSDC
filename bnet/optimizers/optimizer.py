from abc import ABC, abstractmethod
from bnet.generators.generator import TaskGraphGenerator
from bnet.task_graph import TaskGraph
from typing import Iterable, Tuple, Generator
from bnet import Network
from bnet.schedulers import Scheduler


class Optimizer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def optimize_iter(self, 
                      networks: Iterable[Network], 
                      task_graph_generator: TaskGraphGenerator, 
                      scheduler: Scheduler) -> Generator[Tuple[Network, float, TaskGraph, Network, float], None, None]:
        pass

    def optimize(self,
                 networks: Iterable[Network], 
                 task_graph_generator: TaskGraphGenerator, 
                 scheduler: Scheduler) -> Tuple[Network, float]:
        for *_, best_network, best_score in self.optimize_iter(networks, task_graph_generator, scheduler):
            pass 
        return best_network, best_score
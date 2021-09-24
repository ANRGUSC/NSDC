from abc import ABC, abstractmethod
from bnet import TaskGraph, Network

class Scheduler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def schedule(self, network: Network, task_graph: TaskGraph) -> float:
        pass 


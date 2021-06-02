from abc import ABC, abstractmethod
from bnet import TaskGraph, Network

class Scheduler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def schedule(self, task_graph: TaskGraph, network: Network) -> float:
        pass 


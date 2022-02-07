from abc import ABC, abstractmethod
from typing import Hashable, Callable
from bnet import TaskGraph, Network

class Scheduler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def schedule(self,
                 network: Network,
                 task_graph: TaskGraph,
                 can_execute: Callable[[Hashable, Hashable], bool] = lambda *_: True) -> float:
        pass


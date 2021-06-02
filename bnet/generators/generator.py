from abc import ABC, abstractmethod
from typing import Any
from heft.heft import schedule_dag

from .. import Network, TaskGraph


class NetworkGenerator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate(self) -> Network:
        pass 
    

class TaskGraphGenerator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate(self) -> TaskGraph:
        pass 



from abc import ABC, abstractmethod
from ..task_graph import TaskGraph

class TaskGraphGenerator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate(self) -> TaskGraph:
        """Generates a Task Graph

        Returns:
            TaskGraph: generated task graph
        """
        pass 



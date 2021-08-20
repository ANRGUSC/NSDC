from abc import ABC, abstractmethod
from typing import List
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

class CycleGenerator(TaskGraphGenerator):
    """Simple Task Graph Generator which just cycles through a finite \
        collection of TaskGraphs 
    """
    def __init__(self, task_graphs: List[TaskGraph]) -> None:
        super().__init__()
        self.task_graphs = task_graphs
        self._i = 0

    def generate(self) -> TaskGraph:
        task_graph = self.task_graphs[self._i]
        self._i = (self._i + 1) % len(self.task_graphs)
        return task_graph
from abc import ABC, abstractmethod
from bnet.task_graph import TaskGraph
from typing import Any, Optional,  Tuple, Generator
from bnet import Network
from dataclasses import dataclass
import pickle 

@dataclass
class Result(ABC):
    last_network: Optional[Network] = None
    last_metrics: Optional[Any] = None
    last_task_graph: Optional[TaskGraph] = None
    best_network: Optional[Network] = None
    best_metrics: Optional[Any] = None
    

class Optimizer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def optimize_iter(self) -> Generator[Result, None, None]:
        """Optimize iterator which yields itermediate results 
    
        Yields:
            Result: result for current iteration
        """
        pass

    def optimize(self) -> Result:
        """Runs optimization and returns best result/metrics
        
        Returns:
            Result: last result in optimization
        """
        *_, result = self.optimize_iter()
        return result
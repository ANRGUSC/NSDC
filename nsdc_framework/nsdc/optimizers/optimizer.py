from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Generator
from nsdc import Network
from dataclasses import dataclass, field

@dataclass
class Result(ABC):
    network: Network
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    

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
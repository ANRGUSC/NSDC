import numpy as np
from ..network import Network
from typing import Callable, Generator, Iterable
from .optimizer import Optimizer, Result

class BruteForceOptimizer(Optimizer):
    def __init__(self, 
                 networks: Iterable[Network], 
                 cost_func: Callable[[Network], Result]) -> None:
        super().__init__()
        self.networks = networks
        self.cost_func = cost_func

    def optimize_iter(self) -> Generator[Result, None, None]:
        best_result = Result(None, np.inf)
        for network in self.networks:
            result = self.cost_func(network)
            yield result 

            if best_result.cost > result.cost:
                best_result = result 
                
        yield best_result
from .optimizer import Result, Optimizer
from ..network.network import Network

from typing import Callable, Generator

class RandomOptimizer(Optimizer):
    def __init__(self, 
                 mother_network: Network,
                 random_subnetwork: Callable[[], Network],   
                 cost_func: Callable[[Network], Result],          
                 n_iterations: int) -> None:
        self.mother_network = mother_network
        self.random_subnetwork = random_subnetwork
        self.cost_func = cost_func 
        self.n_iterations = n_iterations

    def optimize_iter(self) -> Generator[Result, None, None]:
        best_result = None
        for i in range(self.n_iterations):
            network = self.random_subnetwork()
            result = self.cost_func(network)
            yield result
            if best_result is None or result.cost < best_result.cost:
                best_result = result 
            
        yield best_result
from bnet.network.network import Network
from typing import Callable, Generator
from .optimizer import Optimizer, Result
import numpy as np

class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(self, 
                 mother_network: Network, 
                 start_network: Network,
                 get_neighbor: Callable[[Network], Network],
                 cost_func: Callable[[Network], Result],
                 n_iterations: int,
                 initial_temperature: float) -> None:
        super().__init__()
        self.mother_network = mother_network
        self.get_neighbor = get_neighbor
        self.cost_func = cost_func
        self.start_network = start_network
        self.n_iterations = n_iterations
        self.initial_temperature = initial_temperature

    def optimize_iter(self) -> Generator[Result, None, None]:
        network = self.start_network
        cur_result = self.cost_func(network)
        best_result = cur_result
        for i in range(self.n_iterations):
            result = self.cost_func(self.get_neighbor(cur_result.network))
            yield result
            if result.cost < best_result.cost: # store best seen no matter what
                best_result = result 
            
            diff = result.cost - cur_result.cost
            t = self.initial_temperature / float(i + 1)
            metropolis = np.exp(-diff / t)

            if diff < 0 or np.random.random() < metropolis: # keep this network
                cur_result = result
            
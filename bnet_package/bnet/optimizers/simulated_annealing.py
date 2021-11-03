from bnet.network.network import Network
from typing import Callable, Generator
from .optimizer import Optimizer, Result
import numpy as np

class ExponentialCool:
    """Exponential cooling function

    Accepts results at an exponentially decreasing rate determined by 
    the current "temperature" and a base to determine the rate of decrease.


    Args:
        cur_round: Current round to get temperature for
        cur_result: Current result
        candidate_result: Result being evaluated. If this functio returns true, candidate_result will become \
            cur_result in the next round.
        initial_temperature: Initial temperature (round 0). Default is 10.
        base: base for exponentially decreasing acceptance rate. Default is 1/e.
    
    Returns:
        float: Temperature for round cur_round
    """
    def __init__(self, initial_temperature: float = 10, base: float = 1/np.e) -> None:
        """Constructor for ExponentialCool
        
        Args:
            initial_temperature: Initial temperature (round 0). Default is 10.
            base: base for exponentially decreasing acceptance rate. Default is 1/e.
        """
        self.initial_temperature = initial_temperature
        self.base = base 
        assert(self.base >= 0 and self.base <= 1)

    def __call__(self, 
                 cur_round: int, 
                 cur_result: Result, 
                 candidate_result: Result) -> bool:
        """Exponential cooling function

        Accepts results at an exponentially decreasing rate determined by 
        the current "temperature" and a base to determine the rate of decrease.

        With an intial temperature T, the current temperature t_i in round i is T/(i+1).
        Then, the result is accepted if it has lower cost than the current result or with a probability of \
        base^(-c/t_i) or 0 if base == 0.

        Args:
            cur_round: Current round to get temperature for
            cur_result: Current result
            candidate_result: Result being evaluated. If this functio returns true, candidate_result will become \
                cur_result in the next round.
        
        Returns:
            bool: True if candidate_result should be accepted and False otherwise.
        """
        cur_temp = self.initial_temperature / float(cur_round + 1)
        cur_diff = candidate_result.cost - cur_result.cost
        return cur_diff < 0 or np.random.random() < self.base ** (cur_diff / cur_temp)


class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(self, 
                 mother_network: Network, 
                 start_network: Network,
                 get_neighbor: Callable[[Network], Network],
                 cost_func: Callable[[Network], Result],
                 n_iterations: int,
                 accept: Callable[[int, Result, Result], bool] = ExponentialCool()) -> None:
        super().__init__()
        self.mother_network = mother_network
        self.get_neighbor = get_neighbor
        self.cost_func = cost_func
        self.start_network = start_network
        self.n_iterations = n_iterations

        self.accept = accept

    def optimize_iter(self) -> Generator[Result, None, None]:
        network = self.start_network
        cur_result = self.cost_func(network)
        best_result = cur_result
        for i in range(self.n_iterations):
            result = self.cost_func(self.get_neighbor(cur_result.network))
            yield result
            if result.cost < best_result.cost: # store best seen no matter what
                best_result = result 
            
            if self.accept(i, cur_result, result):
                cur_result = result 

        yield best_result
            
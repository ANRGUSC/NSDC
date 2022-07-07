from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import random
from typing import Callable, Dict, Hashable, Iterable, Optional

from nsdc.network.simple_network import SimpleNetwork
import numpy as np
from itertools import combinations

@dataclass
class Node:
    speed: SimpleNetwork.Speed
    pos: Optional[float] = None

def random_move(max_move: float, x: float) -> float:
    return max(0.0, min(1.0, x + np.random.random() * 2 * max_move - max_move))

class PlaneNetworkFamily:
    def __init__(self, 
                 satellite_bandwidth: Callable[[Iterable[float]], SimpleNetwork.Speed],
                 radio_bandwidth: Callable[[Iterable[float], Iterable[float]], SimpleNetwork.Speed],
                 gray_bandwidth: Callable[[Iterable[float], Iterable[float]], SimpleNetwork.Speed],
                 
                 node_speed: Dict[SimpleNetwork.Speed, float] = {},
                 radio_speed: Dict[SimpleNetwork.Speed, float] = {},
                 sat_speed: Dict[SimpleNetwork.Speed, float] = {},
                 gray_speed: Dict[SimpleNetwork.Speed, float] = {},
                 
                 node_cost: Dict[SimpleNetwork.Speed, float] = {},
                 radio_cost: Dict[SimpleNetwork.Speed, float] = {},
                 sat_cost: Dict[SimpleNetwork.Speed, float] = {},
                 gray_cost: Dict[SimpleNetwork.Speed, float] = {}) -> None:
        super().__init__()
        self.nodes: Dict[Hashable, Node] = {}
        self.satellite_bandwidth = satellite_bandwidth
        self.radio_bandwidth = radio_bandwidth
        self.gray_bandwidth = gray_bandwidth

        self.node_speed = node_speed
        self.radio_speed = radio_speed
        self.sat_speed = sat_speed
        self.gray_speed = gray_speed
        self.node_cost = node_cost
        self.radio_cost = radio_cost
        self.sat_cost = sat_cost
        self.gray_cost = gray_cost

    def add_node(self, 
                 name: Hashable,
                 speed: SimpleNetwork.Speed,
                 pos: Optional[float] = None):
        self.nodes[name] = Node(speed, pos)

    def build_network(self, 
                      nodes: Dict[Hashable, Node] = {}) -> SimpleNetwork:
        network = SimpleNetwork(
            self.node_speed, self.radio_speed, self.sat_speed, self.gray_speed,
            self.node_cost, self.radio_cost, self.sat_cost, self.gray_cost
        )

        for name, node in nodes.items():
            network.add_node(name, node.speed, node.pos)
            sat_speed = self.satellite_bandwidth(node.pos)
            if sat_speed and sat_speed != SimpleNetwork.Speed.NONE:
                network.add_satellite_edge(name, sat_speed)

        for n1, n2 in combinations(nodes.keys(), r=2):
            radio_speed = self.radio_bandwidth(nodes[n1].pos, nodes[n2].pos)
            if radio_speed and radio_speed != SimpleNetwork.Speed.NONE:
                network.add_radio_edge(n1, n2, radio_speed)
            
            gray_speed = self.gray_bandwidth(nodes[n1].pos, nodes[n2].pos)
            if gray_speed and gray_speed != SimpleNetwork.Speed.NONE:
                network.add_gray_edge(n1, n2, gray_speed)
            
        return network

    def random_network(self) -> SimpleNetwork:
        nodes: Dict[Hashable, Node] = {}
        for name, node in self.nodes.items():
            nodes[name] = deepcopy(node)
            if nodes[name].pos is None:
                nodes[name].pos = tuple(np.random.random(2))
        
        return self.build_network(nodes)
            
    def random_neighbor(self, network: SimpleNetwork) -> SimpleNetwork:
        nodes: Dict[Hashable, Node] = {}
        moving_node = random.choice([
            name for name, node in self.nodes.items() 
            if node.pos is None
        ])
            
        for name in network.nodes:
            pos = network.get_pos(name)
            if name == moving_node:
                pos = tuple(map(partial(random_move, 0.1), pos))

            nodes[name] = Node(speed=network.get_speed(name), pos=pos)
            if nodes[name].pos is None:
                nodes[name].pos = tuple(np.random.random(2))
        
        return self.build_network(nodes)

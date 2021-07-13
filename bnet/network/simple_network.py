from bnet import network
from itertools import chain, combinations
from typing import Callable, Dict, List, Optional, Set, Tuple, Generator, Union
import networkx as nx
import numpy as np
import pandas as pd 
from enum import Enum
from networkx.generators.geometric import random_geometric_graph
from numpy.random import zipf, rand
import random
from .network import Network
import matplotlib.pyplot as plt

class SimpleNetwork(Network):
    class Speed(Enum):
        NONE = 0
        LOW = 1
        HIGH = 2
        
    SpeedConverter = Optional[
        Union[
            Callable[["SimpleNetwork.Speed"], float], 
            Dict["SimpleNetwork.Speed", float]
        ]
    ]
    
    def __init__(self, 
                 node_speed: Dict["SimpleNetwork.Speed", float] = {},
                 radio_speed: Dict["SimpleNetwork.Speed", float] = {},
                 sat_speed: Dict["SimpleNetwork.Speed", float] = {},
                 gray_speed: Dict["SimpleNetwork.Speed", float] = {}) -> None:
        """Constructor for SimpleNetwork

        Args:
            node_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                node speed (float). By Default, NONE=0, LOW=1, and HIGH=2.
            radio_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                bandwidth (float) for radio edges. By Default, NONE=0, LOW=1, and HIGH=2.
            sat_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                bandwidth (float) for satellite connection. By Default, NONE=0, LOW=1, and HIGH=2.
            gray_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                bandwidth (float) gray network edges. By Default, NONE=0, LOW=1, and HIGH=2.
        """
        super().__init__()
        self._graph = nx.MultiGraph()
        self.node_speed = {
            speed: node_speed.get(speed, speed.value)
            for speed in list(SimpleNetwork.Speed)
        }
        self.radio_speed = {
            speed: radio_speed.get(speed, speed.value)
            for speed in list(SimpleNetwork.Speed)
        }
        self.sat_speed = {
            speed: sat_speed.get(speed, speed.value)
            for speed in list(SimpleNetwork.Speed)
        }
        self.gray_speed = {
            speed: gray_speed.get(speed, speed.value)
            for speed in list(SimpleNetwork.Speed)
        }

    @property
    def nodes(self) -> List[str]:
        """List of nodes in the network"""
        return [node for node in self._graph.nodes if node != "__satellite__"]

    @property
    def edges(self) -> List[Tuple[str, str, str]]:
        """List of edges in the network"""
        return list(self._graph.edges)

    def cost(self) -> float:
        cost = 0.0
        for node in self._graph.nodes:
            if node == "__satellite__":
                continue
            cost += self.node_speed[self._graph.nodes[node]["speed"]]
        for edge in self._graph.edges:
            _, _, key = edge
            if key == "satellite":
                cost += self.sat_speed[self._graph.edges[edge]["speed"]]
            elif key == "radio":
                cost += self.radio_speed[self._graph.edges[edge]["speed"]]
            else:
                cost += self.gray_speed[self._graph.edges[edge]["speed"]]
        
        return cost 

    def add_node(self, name: str, speed: "SimpleNetwork.Speed", pos: Tuple[float, float]) -> None:
        """Adds a node to the graph

        Args:
            name: Name of the node to add.
            speed: Speed of the node to add.
            pos: Position of the node in the graph on the plane (in the unit square).
        """
        if not all(map(lambda x: x >= 0 and x <= 1, pos)):
            raise ValueError("Position must be in unit hypercube")
        self._graph.add_node(name, speed=speed, pos=pos)

    def remove_node(self, name: str) -> None:
        """Removes a node from the network

        Args:
            name: name of node to remove
        """
        self._graph.remove_node(name)
    
    def add_edge(self, src: str, dst: str, speed: "SimpleNetwork.Speed", key: Optional[str] = None) -> None:
        """Adds an edge to the graph.

        This method should not be called directly. Instead, users should use the add_radio_edge,
        add_gray_edge, and add_satellite_edge methods.

        Args:
            src: Source of edge to remove.
            dst: Destination of edge to remove.
            speed: Speed of the edge (NONE, LOW, HIGH).
            key: key of edge (either "satellite", "radio", or "gray")
        """
        assert(key in {"satellite", "radio", "gray"})
        self._graph.add_edge(src, dst, key=key, speed=speed)

    def remove_edge(self, src: str, dst: str, key: Optional[str] = None) -> None:
        """Removes an edge from the graph

        Args:
            src: Source of edge to remove.
            dst: Destination of edge to remove.
            key: key of edge (either "satellite", "radio", or "gray")
        """
        assert(key in {"satellite", "radio", "gray"})
        self._graph.remove_edge(src, dst, key=key)

    def add_radio_edge(self, src: str, dst: str, speed: "SimpleNetwork.Speed") -> None:
        """Adds a radio edge between two nodes in the graph

        Args:
            src: Source of edge to remove.
            dst: Destination of edge to remove.
            speed: Speed of the edge (NONE, LOW, HIGH).
        """
        if speed != SimpleNetwork.Speed.NONE:
            self.add_edge(src, dst, speed, key="radio")

    def add_gray_edge(self, src: str, dst: str, speed: "SimpleNetwork.Speed") -> None:
        """Adds a gray network edge between two nodes in the graph

        Args:
            src: Source of edge to remove.
            dst: Destination of edge to remove.
            speed: Speed of the edge (NONE, LOW, HIGH).
        """
        if speed != SimpleNetwork.Speed.NONE:
            self.add_edge(src, dst, speed, key="gray")  

    def add_satellite_edge(self, node: str, speed: "SimpleNetwork.Speed") -> None:
        """Adds a satellite edge to a node in the graph

        Args:
            node: Node to add satellite connectio to.
            speed: Speed of the edge (NONE, LOW, HIGH).
        """
        if speed != SimpleNetwork.Speed.NONE:
            self.add_edge(node, "__satellite__", speed, key="satellite")  

    def get_radio_edge(self, src: str, dst: str) -> "SimpleNetwork.Speed":
        """Gets radio edge speed

        Returns:
            SimpleNetwork.Speed: speed of radio edge between nodes src and dst (NONE, LOW, HIGH). \
                If no edge exists, the NONE speed is returned.
        """
        return self._graph.get_edge_data(src, dst, key="radio", default={}).get("speed", SimpleNetwork.Speed.NONE)

    def get_gray_edge(self, src: str, dst: str) -> "SimpleNetwork.Speed":
        """Gets gray network edge speed

        Returns:
            SimpleNetwork.Speed: speed of gray network edge between nodes src and dst (NONE, LOW, HIGH). \
                If no edge exists, the NONE speed is returned.
        """
        return self._graph.get_edge_data(src, dst, key="gray", default={}).get("speed", SimpleNetwork.Speed.NONE)

    def get_satellite_edge(self, node: str) -> "SimpleNetwork.Speed":
        """Gets satellite connection speed of node

        Returns:
            SimpleNetwork.Speed: speed of satellite connection of node (NONE, LOW, HIGH). \
                If no satellite connection exists, the NONE speed is returned.
        """
        return self._graph.get_edge_data(node, "__satellite__", key="satellite", default={}).get("speed", SimpleNetwork.Speed.NONE)

    def bandwidth_matrix(self) -> pd.DataFrame:
        """Returns a pandas DataFrame of the node-to-node bandwidth

        Returns:
            pd.DataFrame: Rows and columns are nodes in the network. Cells are \
                bandwidth between nodes in the network.
        """
        graph = self.to_networkx()
        return pd.DataFrame(
            [
                [
                    np.inf if n1 == n2 else graph.get_edge_data(n1, n2, {}).get("speed", 0)
                    for n2 in graph.nodes
                ]
                for n1 in graph.nodes
            ],
            columns=graph.nodes,
            index=graph.nodes
        )

    def to_networkx_multigraph(self) -> nx.MultiGraph:
        graph = self._graph.copy() 
        nx.set_node_attributes(
            graph, 
            {node: self.node_speed[self._graph.nodes[node]["speed"]] for node in self._graph.nodes}, 
            name="speed"
        )
        speeds = {}
        for src, dst, key in self._graph.edges:
            speed = self._graph.edges[(src, dst, key)]["speed"]
            if key == "satellite":
                speeds[(src, dst, key)] = self.sat_speed[speed]
            elif key == "radio":
                speeds[(src, dst, key)] = self.radio_speed[speed]
            else:
                speeds[(src, dst, key)] = self.gray_speed[speed]
        nx.set_edge_attributes(graph, speeds, name="speed")

    def to_networkx(self) -> nx.Graph:
        """Converts the network into a networkx representation.

        The networkx graph should have "speed" node attribute and "bandwidth" edge attribute.

        Returns:
            nx.Graph: NetworkX representation of network.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        nx.set_node_attributes(
            graph, 
            {
                node: self.node_speed[self._graph.nodes[node]["speed"]] 
                for node in self._graph.nodes
            }, 
            name="speed"
        )
        
        nx.set_node_attributes(
            graph, 
            nx.get_node_attributes(self._graph, "pos"), 
            name="pos"
        )
        
        for src, dst in combinations(self._graph.nodes, r=2):
            if src == "__satellite__" or dst == "__satellite__":
                continue 
              
            sat_speed = min(
                self.sat_speed[self.get_satellite_edge(dst)],
                self.sat_speed[self.get_satellite_edge(src)]
            )
            gray_speed = self.gray_speed[self.get_gray_edge(src, dst)]
            radio_speed = self.radio_speed[self.get_radio_edge(src, dst)] 

            graph.add_edge(src, dst, speed=sat_speed + gray_speed + radio_speed)
        
        return graph

    def iter_subnetworks(self, min_size: int = 3) -> Generator["SimpleNetwork", None, None]:
        """Iterates over subnetworks of the network

        Args:
            min_size: Minimum size of subnetworks to yield.
        Yields:
            SimpleNetwork: Network with a subset of the edges.
        """
        edges = list(self._graph.edges)
        for subedges in chain.from_iterable(combinations(edges, r) for r in range(min_size, len(edges) + 1)):
            network = SimpleNetwork(self.node_speed, self.radio_speed, self.sat_speed, self.gray_speed)
            network._graph = self._graph.edge_subgraph(subedges)
            yield network 

    def draw(self, subedges: Set, ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        satellite_subedges = {
            a if b == "__satellite__" else b 
            for a, b, key in subedges
            if key == "satellite"
        }
        graph: nx.MultiGraph = self._graph.copy()
        graph.remove_node("__satellite__")
        colors = {
            SimpleNetwork.Speed.NONE: "gray", 
            SimpleNetwork.Speed.LOW: "#88C0D0",
            SimpleNetwork.Speed.HIGH: "#A3BE8C"
        }
        styles = {
            "gray": "solid", # "arc3, rad = 0.2",
            "radio": "dotted", # "arc3, rad = 0.0"
        }
        widths = {
            "gray": 3, # "arc3, rad = 0.2",
            "radio": 6, # "arc3, rad = 0.0"
        }
        node_colors = [
            colors[self.get_satellite_edge(node)]
            for node in graph.nodes
        ]
        node_alpha = [
            1 if node in satellite_subedges else 0.2
            for node in graph.nodes
        ]
        edge_styles = [
            styles[key] for src, dst, key in graph.edges
        ]
        edge_widths = [
            widths[key] for src, dst, key in graph.edges
        ]
        edge_alpha = [
            1 if edge in subedges else 0.2
            for edge in graph.edges
        ]
        edge_colors = [
            colors[self.get_gray_edge(src, dst) if key == "gray" else self.get_radio_edge(src, dst)]
            for src, dst, key in graph.edges
        ]
        pos = [
            graph.nodes[node]["pos"]
            for node in graph.nodes
        ]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        nx.draw_networkx_nodes(
            graph, pos=pos, ax=ax,
            alpha=node_alpha, 
            node_color=node_colors
        )
        nx.draw_networkx_labels(
            graph, pos=pos, ax=ax
        )
        nx.draw_networkx_edges(
            graph, pos=pos, ax=ax,
            edge_color=edge_colors, style=edge_styles, width=edge_widths,
            alpha=edge_alpha
        )
        return fig, ax
        

    @classmethod
    def random_zipf(cls, 
                    zipf_constant: float,
                    node_speed: Dict["SimpleNetwork.Speed", float] = None,
                    radio_speed: Dict["SimpleNetwork.Speed", float] = None,
                    sat_speed: Dict["SimpleNetwork.Speed", float] = None,
                    gray_speed: Dict["SimpleNetwork.Speed", float] = None) -> "SimpleNetwork":
        """Generates a random SimpleNetwork

        Generates a random geometric graph (in the unit square) with a random radius (0 to 1) to determine \
        node position and radio edges (each with uniformly random speed either LOW or HIGH). \
        A random threshold x (0 to 1) is chosen so that every node whose x-coordinate is less than x shares \
        a gray edge (with uniformly random speed either LOW or HIGH) with every other node that satisfies this \
        condition. \
        Finally, each nodes is given a uniformly random satellite connection chosen uniformly at random from \
        NONE, LOW, or HIGH).
        
        Args:
            zipf_constant: Zipf constant for generating networks with different sizes. 
            node_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                node speed (float). By Default, NONE=0, LOW=1, and HIGH=2.
            radio_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                bandwidth (float) for radio edges. By Default, NONE=0, LOW=1, and HIGH=2.
            sat_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                bandwidth (float) for satellite connection. By Default, NONE=0, LOW=1, and HIGH=2.
            gray_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                bandwidth (float) gray network edges. By Default, NONE=0, LOW=1, and HIGH=2.
        """
        return cls.random(
            num_nodes=zipf(zipf_constant),
            node_speed=node_speed,
            radio_speed=radio_speed,
            sat_speed=sat_speed,
            gray_speed=gray_speed
        )

    @classmethod
    def random(cls, 
               num_nodes: float,
               node_speed: Dict["SimpleNetwork.Speed", float] = None,
               radio_speed: Dict["SimpleNetwork.Speed", float] = None,
               sat_speed: Dict["SimpleNetwork.Speed", float] = None,
               gray_speed: Dict["SimpleNetwork.Speed", float] = None) -> "SimpleNetwork":
        """Generates a random SimpleNetwork

        Generates a random geometric graph (in the unit square) with a random radius (0 to 1) to determine \
        node position and radio edges (each with uniformly random speed either LOW or HIGH). \
        A random threshold x (0 to 1) is chosen so that every node whose x-coordinate is less than x shares \
        a gray edge (with uniformly random speed either LOW or HIGH) with every other node that satisfies this \
        condition. \
        Finally, each nodes is given a uniformly random satellite connection chosen uniformly at random from \
        NONE, LOW, or HIGH).
        
        Args:
            num_nodes: Number of nodes in graph. 
            node_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                node speed (float). By Default, NONE=0, LOW=1, and HIGH=2.
            radio_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                bandwidth (float) for radio edges. By Default, NONE=0, LOW=1, and HIGH=2.
            sat_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                bandwidth (float) for satellite connection. By Default, NONE=0, LOW=1, and HIGH=2.
            gray_speed: Function that converts a SimpleNetwork.Speed (NONE, LOW, HIGH) to an actual \
                bandwidth (float) gray network edges. By Default, NONE=0, LOW=1, and HIGH=2.
        """
        graph: nx.Graph = random_geometric_graph(num_nodes, rand())
        pos = nx.get_node_attributes(graph, "pos")
        network = cls(node_speed, radio_speed, sat_speed, gray_speed)

        rand_speed = lambda: random.choice(list(SimpleNetwork.Speed))
        rand_pos_speed = lambda: random.choice([SimpleNetwork.Speed.LOW, SimpleNetwork.Speed.HIGH])

        # Add satellite node 
        network.add_node("__satellite__", SimpleNetwork.Speed.NONE, pos=(0, 0))
        for node, p in pos.items():
            network.add_node(node, speed=rand_pos_speed(), pos=p)
            network.add_edge(node, "__satellite__", rand_speed(), key="satellite")
        
        # Add radio edges
        for src, dst in graph.edges:
            network.add_edge(src, dst, rand_pos_speed(), key="radio")

        # Add gray network edges 
        gray_threshold = rand()
        for src, dst in combinations(graph.nodes, r=2):
            if pos[src][0] < gray_threshold and pos[dst][0] < gray_threshold:
                network.add_edge(src, dst, rand_pos_speed(), key="gray")

        return network
    

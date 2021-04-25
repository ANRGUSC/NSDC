import networkx as nx 
from typing import Callable, Iterable, Generator, TypeVar, Dict, List, Any
from itertools import product, chain, combinations
from networkx.algorithms.triads import all_triads

from numpy import random

def all_graphs(n: int) -> Generator[nx.Graph, None, None]:
    possible_edges = [(i, j) for i in range(n) for j in range(i)]
    for edges in chain.from_iterable(combinations(possible_edges, r) for r in range(len(possible_edges)+1)):
        graph = nx.Graph()
        graph.add_edges_from(edges)
        if nx.is_connected(graph):
            yield graph 

def all_dags(n: int) -> Generator[nx.DiGraph, None, None]:
    possible_edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    for edges in chain.from_iterable(combinations(possible_edges, r) for r in range(1, len(possible_edges)+1)):
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        if nx.is_directed_acyclic_graph(graph):
            yield graph 

def all_augmented_graphs(n: int, 
                         dag: bool,
                         node_attributes: Dict[str, List[Any]], 
                         edge_attributes: Dict[str, List[Any]]) -> Generator[nx.DiGraph, None, None]:
    node_attr_names, node_attr_values = zip(*node_attributes.items())
    node_attr_combos = [dict(zip(node_attr_names, attrs)) for attrs in product(*node_attr_values)]
    edge_attr_names, edge_attr_values = zip(*edge_attributes.items())
    edge_attr_combos = [dict(zip(edge_attr_names, attrs)) for attrs in product(*edge_attr_values)]

    for graph in (all_dags if dag else all_graphs)(n):
        node_names = list(graph.nodes)
        edge_names = list(graph.edges)
        for node_attrs in product(*[node_attr_combos for _ in range(graph.order())]):
            for edge_attrs in product(*[edge_attr_combos for _ in range(graph.size())]):
                _graph = graph.copy()
                nx.set_node_attributes(_graph, dict(zip(node_names, node_attrs)))
                nx.set_edge_attributes(_graph, dict(zip(edge_names, edge_attrs)))
                yield _graph 

def zipf(N: int, zipf_constant: float) -> List[float]:
    denom = sum([1/(n**zipf_constant) for n in range(1, N + 1)])
    return [(1/(i**zipf_constant))/denom for i in range(1, N + 1)]

AnyGraph = TypeVar('AnyGraph', bound=nx.Graph)
def get_graph_sampler(graphs: Iterable[AnyGraph], 
                      metric: Callable[[AnyGraph], float],
                      zipf_constant: float) -> Callable[[], AnyGraph]:
    sorted_graphs = sorted(graphs, key=lambda graph: metric(graph))
    print("size: ", len(sorted_graphs))
    dist = zipf(len(sorted_graphs), zipf_constant)
    return lambda: random.choice(sorted_graphs, p=dist)

def main():
    network_order = 3
    task_order = 5

    network_sampler = get_graph_sampler(
        all_augmented_graphs(
            network_order,
            dag=False,
            node_attributes={
                "mem": [0, 1, 2],
                "comp": [0, 1, 2],
                "out": [0, 1, 2] 
            }
        ),
        metric=lambda x: -1
    ) 

    task_sampler = get_graph_sampler(
        all_augmented_graphs(
            task_order,
            dag=True,
            node_attributes={
                "mem": [0, 1, 2],
                "comp": [0, 1, 2]
            },
            edge_attributes={
                "band": [0, 1, 2]
            }
        ),
        metric=lambda x: 1,
        zipf_constant=1
    ) 

if __name__ == "__main__":
    main()
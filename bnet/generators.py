from typing import Tuple
import networkx as nx 
import random 

def rand(low: float, high: float) -> float:
    return random.random() * (high - low) + low 

def random_network(num_nodes: int, 
                   comp_multiplier_range: Tuple[float, float], 
                   comm_range: Tuple[float, float]) -> nx.Graph:
    graph: nx.Graph = nx.complete_graph(num_nodes)
    nx.set_node_attributes(graph, {node: rand(*comp_multiplier_range) for node in graph.nodes}, name="comp_multiplier")
    nx.set_edge_attributes(graph, {edge: rand(*comm_range) for edge in graph.edges}, name="weight")
    return graph 

def random_task_graph(num_tasks: int, 
                      comp_range: Tuple[float, float], 
                      data_range: Tuple[float, float]) -> nx.DiGraph:
    graph = nx.complete_graph(num_tasks)
    dag = nx.DiGraph([
        (u, v, {"weight": rand(*data_range)}) 
        for (u,v) in graph.edges() if u<v
    ])

    nx.set_node_attributes(dag, {node: rand(*comp_range) for node in dag.nodes}, name="comp")

    return dag 
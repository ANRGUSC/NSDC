"""
DAG generation

"""

"""
https://ipython.org/ipython-doc/stable/parallel/dag_dependencies.html

"""
import numpy as np
import networkx as nx
import random as ran
import matplotlib.pyplot as plt
import heft
import core as cor
import Network_N

#-------------------
def random_dag(nodes, edges):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""

    disconnected =0

    while disconnected == 0:
        
        G = nx.DiGraph()
        temp_edges=edges
        
        for i in range(nodes):
            G.add_node(i)
        
        
        while temp_edges > 0:
            a = ran.randint(0,nodes-1)
            b=a
            while b==a:
                b = ran.randint(0,nodes-1)
            G.add_edge(a,b)
            if nx.is_directed_acyclic_graph(G):
                temp_edges -= 1
            else:
                # we closed a loop!
                G.remove_edge(a,b)
        #if nx.is_weakly_connected(G):
        disconnected=1

    
    
    dag_dict = {}
    
    for node in G:
        dag_dict[node] = [n for n in G.neighbors(node)]
    
    dag=dag_dict
    
    return G,dag

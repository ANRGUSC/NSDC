"""
DAG generation

"""

"""
https://ipython.org/ipython-doc/stable/parallel/dag_dependencies.html

"""
from copy import deepcopy
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
    nnodes=deepcopy(nodes)
    while disconnected == 0:
        
        G = nx.DiGraph()
        temp_edges=deepcopy(edges)
        
        for i in range(nnodes):
            G.add_node(i)
        
        
        while temp_edges > 0:
            a = ran.randint(0,nnodes-1)
            b=a
            while b==a:
                b = ran.randint(0,nnodes-1)
            G.add_edge(a,b)
            if nx.is_directed_acyclic_graph(G):
                temp_edges -= 1
            else:
                # we closed a loop!
                G.remove_edge(a,b)
        #if nx.is_weakly_connected(G):
        disconnected=1
        
        
        #plt.figure(0)
        #nx.draw(G)
    
    
    dag_dict = {}
    
    for node in G:
        dag_dict[node] = [n for n in G.neighbors(node)]
    
    dag=dag_dict
    
    return G,dag

"""
# Main

G_dag,dag=random_dag(10, 15)
plt.figure(0)
nx.draw(G_dag)
"""
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
def random_dag(nodes, scale):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""

    #disconnected =0
    #nnodes=deepcopy(nodes)
    if scale == 'n':
        
        G = nx.DiGraph()
        #temp_nodes=deepcopy(nodes)
        
        for i in range(nodes):
            G.add_node(i)
        
        
        for j in range(0,int(nodes/2)):
            G.add_edge(0,j+1)
        
        for i in range(1,int(nodes/2)):
            G.add_edge(i,nodes/2+i)
        G.add_edge(nodes/2,nodes-1)
        
        
        #if nx.is_weakly_connected(G):
        #disconnected=1
    
    if scale == 'n2':  
        G = nx.DiGraph()
        
        for i in range(nodes):
            G.add_node(i)        
        
        index=1
        
        for i in range(0,nodes):
            
            
            
            for j in range(index,nodes):
            
                G.add_edge(i,j)        
            
            index +=1
    
    
    
    
    dag_dict = {}
    
    for node in G:
        dag_dict[node] = [n for n in G.neighbors(node)]
    
    dag=dag_dict
    
    return G,dag


# Main

"""
scale='n2'
G_dag,dag=random_dag(5, scale)
plt.figure(0)
#nx.draw(G_dag)
pos = nx.spring_layout(G_dag)  # Seed layout for reproducibility
nx.draw(G_dag, pos=pos, with_labels=True)
"""

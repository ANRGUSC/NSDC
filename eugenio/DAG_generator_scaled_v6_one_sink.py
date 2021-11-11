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
#import heft
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
        
        
        for j in range(0,round(int(nodes/2))):
            G.add_edge(0,j+1,weight=ran.randrange(1,5))
        
        for i in range(1,round(int(nodes/2))):
            G.add_edge(i,round(nodes/2)+i,weight=ran.randrange(1,5))
        
        G.add_edge(round(nodes/2),nodes-1,weight=ran.randrange(1,5))
        
        
        #if nx.is_weakly_connected(G):
        #disconnected=1
    
    if scale == 'n2':  
        G = nx.DiGraph()
        
        for i in range(nodes):
            G.add_node(i)        
        
        index=1
        
        for i in range(0,nodes):
            
            
            
            for j in range(index,nodes):
            
                G.add_edge(i,j,weight=ran.randrange(1,5))        
            
            index +=1
    
    
    #We prepare the DAG to have only one sink and one source:
    #--------------
    G.add_node(len(G.nodes()))# add sink
    nodes_numb=len(G.nodes())

    for i in range(1,len(G.nodes())-1):
        if G.out_degree(i) == 0:
            G.add_edge(i,nodes_numb-1,weight=0)

    # Here we try to fix the cases when "nodes" is odd; thus, we try to get
    # the desired o(n) edges 
    
    # if theres is a "source like" node we add it to node o (zero); else,
    # we just add and edge to complete the O(n) edges:
    flag=0
    for i in range(1,len(G.nodes())):
        if G.in_degree(i) == 0:
            G.add_edge(0,i,weight=ran.randrange(1,5))
            flag=1
    if flag != 1:
        G.add_edge(0,nodes-1,weight=ran.randrange(1,5))
            
    """
    G.add_node(len(G.nodes()))#add source
    nodes_numb=len(G.nodes())
    

    """
    """
    ### Relabel of nodes: #####
    nodes_numb=len(G.nodes())
    for i in range(0,nodes_numb-1):
        mapping={i:str(i+1)}
        nx.relabel_nodes(G, mapping, copy=False)
    mapping={nodes_numb-1:"0"}
    nx.relabel_nodes(G, mapping, copy=False)   
    #################
    """





    #plt.figure(1)
    #nx.draw(G)
    #-----------------
    
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




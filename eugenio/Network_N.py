import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as py
import more_itertools as combinatorials
from itertools import combinations



def subgraphs(N,min_N):

    # N is the total number of nodes of the main Graph
    

    stored_list=[]
    index=np.arange(min_N,N) # we consider min_N as the minimum subgraph 
    # number of nodes
    
    for s in range(1,max(index.shape)+1):
    
        #comb = combinations(range(1,N+1), index[s-1])
        comb = combinations(range(0,N), index[s-1])
        for i in list(comb):
        	stored_list.append(i)
    
    
    
    G = nx.complete_graph(N)
    H_list=[] # H will be the different subgraphs of G
    
    for i in range(1,len(stored_list)+1):
        
        H_list.append(G.subgraph(stored_list[i-1]))
    

    

    return G,H_list


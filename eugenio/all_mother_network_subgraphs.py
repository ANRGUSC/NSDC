import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as py
import more_itertools as combinatorials
from itertools import combinations
from itertools import product
import Network_N


# This module returns the connectivity of the main mother network "G" and all the possible 
# subgraphs of the mother network. 

class all_subgraphs: # list with each subgraph of the mother network N and each possible 
                     # connectivity choice of edges
                       
    def __init__(self, N_subgraph, edge_weight_choices): 
        self.N_subgraph = N_subgraph
        self.edge_weight_choices = edge_weight_choices
           
    def total_len(self):
        tot_len=len(self.edge_weight_choices)
        return tot_len
    

def all_mother_network_subgraphs(N,min_N,Connectivity_Matrix):
    
    G,H_list = Network_N.subgraphs(N, min_N) # we send N and ,min_N to the module Network_N.
    
    H_list_connectivities=[]
    
    for i in range(0,len(H_list)):
    
        edges_=list(H_list[i].edges)
        list_=[]
        temp_sum=0
        temp_list=[]
        results=[]
        def_results=[]
        
        for j in range(0,len(edges_)):
            temp_sum+=Connectivity_Matrix[edges_[j][0]][edges_[j][1]]
            temp_list.append(Connectivity_Matrix[edges_[j][0]][edges_[j][1]])
            
        list_=list(range(int(temp_sum)))   
        for c in product(list_, repeat = len(edges_)):
            results.append(c)
    
        
        # Here we detect a wrong permutation choice: one that has an incorrrect
        # RF/gray/ satellite assignment. Example: we have 3 edges, each one of them 
        # only has an RF connection; thus, a permutation result(1,1,1) is not 
        # possible because that would mean that each edge could have a pair from 
        # RF/gray/satellite (0 and 1) choice.
        
        for s in range(0,len(results)):
            flag=len(edges_)
            for t in range(0,len(edges_)):
                if results[s][t] < temp_list[t]:
                    flag -= 1
    
                    
            if flag == 0: 
                def_results.append(results[s])
                
        H_list_connectivities.append(all_subgraphs(H_list[i],def_results))

    return G,H_list_connectivities
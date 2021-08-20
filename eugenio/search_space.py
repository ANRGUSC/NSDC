# This script creates the search space for the "simulated annelaing" software


    
from copy import deepcopy
import numpy as np
#import networkx as nx
#import random as ran
#import matplotlib.pyplot as plt
#import heft
import core as cor
#import Network_N
#import DAG_generator
#import more_itertools
#from itertools import product
#from itertools import combinations
#import all_mother_network_subgraphs



class all_subgraphs: # list with each subgraph of the mother network N and each possible 
                     # connectivity choice of edges
                       
    def __init__(self, N_subgraph, edge_weight_choices): 
        self.N_subgraph = N_subgraph
        self.edge_weight_choices = edge_weight_choices
           
    def total_len(self):
        tot_len=len(self.edge_weight_choices)
        return tot_len
    


"""
    def __init__(self,H_list_connectivities,Connectivity_Matrix,Network_N_matrix,nodes,bandwith_table,user_input): 
        self.H_list_connectivities=H_list_connectivities
        self.Connectivity_Matrix=Connectivity_Matrix
        self.Network_N_matrix=Network_N_matrix
        self.nodes=nodes
        self.bandwith_table=bandwith_table
        self.user_input=user_input
""" 
    
    
def search_space(H_list_connectivities,Connectivity_Matrix,Network_N_matrix,nodes,bandwith_table,user_input):
    
    


    
    search_space_choices=[]
    
    number_of_subgraphs=[]
    temp_H_list=[]
    
    
    for obj in H_list_connectivities:
       
       number_of_subgraphs.append(obj.total_len())
       for i in range(0,obj.total_len()):
           temp_H_list.append(all_subgraphs(obj.N_subgraph, obj.edge_weight_choices[i]))
    
    H_list=[]
    for obj in temp_H_list:
        H_list.append(obj.N_subgraph)
    


    
    
    for r in range(0,sum(number_of_subgraphs)): # different Mother Network's subgraphs loop
        
            # Here we fill up the Network_N_matrix with the different bandwiths
            # according to each subgraph edge choice (RF/Gray/Satellite)
            
            Network_N_matrixb=deepcopy(Network_N_matrix)
            
            edges_=list(H_list[r].edges)
            for j in range(0,len(edges_)):
                
                #ONE EDGE
                if (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 1 ) and (user_input[edges_[j][0]][edges_[j][1]] == 1):#RF
                      
                    Network_N_matrixb[edges_[j][0]][edges_[j][1]]=bandwith_table[0]
                    Network_N_matrixb[edges_[j][1]][edges_[j][0]]=Network_N_matrixb[edges_[j][0]][edges_[j][1]]# we make it symmetric
                    
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 1 ) and (user_input[edges_[j][0]][edges_[j][1]] == 2):#gray:
            
                    Network_N_matrixb[edges_[j][0]][edges_[j][1]]=bandwith_table[1]
                    Network_N_matrixb[edges_[j][1]][edges_[j][0]]=Network_N_matrixb[edges_[j][0]][edges_[j][1]]
        
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 1 ) and (user_input[edges_[j][0]][edges_[j][1]] == 3):#satellite:
        
                    Network_N_matrixb[edges_[j][0]][edges_[j][1]]=bandwith_table[2]
                    Network_N_matrixb[edges_[j][1]][edges_[j][0]]=Network_N_matrixb[edges_[j][0]][edges_[j][1]]
        
                #TWO EDGES
                
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 2 ) and (user_input[edges_[j][0]][edges_[j][1]] == 4):#RF/gray:
        
                    Network_N_matrixb[edges_[j][0]][edges_[j][1]]=bandwith_table[temp_H_list[r].edge_weight_choices[j]]
                    Network_N_matrixb[edges_[j][1]][edges_[j][0]]=Network_N_matrixb[edges_[j][0]][edges_[j][1]]
                
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 2 ) and (user_input[edges_[j][0]][edges_[j][1]] == 5):#RF/satellite:
        
                    Network_N_matrixb[edges_[j][0]][edges_[j][1]]=bandwith_table[temp_H_list[r].edge_weight_choices[j] * 2]
                    Network_N_matrixb[edges_[j][1]][edges_[j][0]]=Network_N_matrixb[edges_[j][0]][edges_[j][1]]
                
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 2 ) and (user_input[edges_[j][0]][edges_[j][1]] == 6):#gray/satellite:
        
                    Network_N_matrixb[edges_[j][0]][edges_[j][1]]=bandwith_table[temp_H_list[r].edge_weight_choices[j] + 1]
                    Network_N_matrixb[edges_[j][1]][edges_[j][0]]=Network_N_matrixb[edges_[j][0]][edges_[j][1]]
                
                
                #THREE EDGES
            
            
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 3 ):#RF/gray/satellite:
        
                    Network_N_matrixb[edges_[j][0]][edges_[j][1]]=bandwith_table[temp_H_list[r].edge_weight_choices[j]]
                    Network_N_matrixb[edges_[j][1]][edges_[j][0]]=Network_N_matrixb[edges_[j][0]][edges_[j][1]]
                
            
            
            
            
            search_space_choices.append(Network_N_matrixb)

            
    return search_space_choices

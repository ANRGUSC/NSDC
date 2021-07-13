import numpy as np
import networkx as nx
import random as ran
import matplotlib.pyplot as plt
import heft
import core as cor
import Network_N
import DAG_generator
import more_itertools
from itertools import product
from itertools import combinations
import all_mother_network_subgraphs


# This function receives all possible subgraphs, all possible tasks and all the user inputs
# and returns the Average Performance and the associated Cost for each subgraph.

class all_subgraphs: # list with each subgraph of the mother network N and each possible 
                     # connectivity choice of edges
                       
    def __init__(self, N_subgraph, edge_weight_choices): 
        self.N_subgraph = N_subgraph
        self.edge_weight_choices = edge_weight_choices
           
    def total_len(self):
        tot_len=len(self.edge_weight_choices)
        return tot_len
    
    
def optimization(H_list_connectivities,Connectivity_Matrix,Network_N_matrix,nodes,bandwith_table,user_input,Cost_Matrix,DAG_matrices,dags):
    
    Average_Performance=[]
    Cost_vector=[]
    
    
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
            
            
            edges_=list(H_list[r].edges)
            for j in range(0,len(edges_)):
                
                #ONE EDGE
                if (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 1 ) and (user_input[edges_[j][0]][edges_[j][1]] == 1):#RF
                      
                    Network_N_matrix[edges_[j][0]][edges_[j][1]]=bandwith_table[0]
                    Network_N_matrix[edges_[j][1]][edges_[j][0]]=Network_N_matrix[edges_[j][0]][edges_[j][1]]# we make it symmetric
                    
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 1 ) and (user_input[edges_[j][0]][edges_[j][1]] == 2):#gray:
            
                    Network_N_matrix[edges_[j][0]][edges_[j][1]]=bandwith_table[1]
                    Network_N_matrix[edges_[j][1]][edges_[j][0]]=Network_N_matrix[edges_[j][0]][edges_[j][1]]
        
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 1 ) and (user_input[edges_[j][0]][edges_[j][1]] == 3):#satellite:
        
                    Network_N_matrix[edges_[j][0]][edges_[j][1]]=bandwith_table[2]
                    Network_N_matrix[edges_[j][1]][edges_[j][0]]=Network_N_matrix[edges_[j][0]][edges_[j][1]]
        
                #TWO EDGES
                
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 2 ) and (user_input[edges_[j][0]][edges_[j][1]] == 4):#RF/gray:
        
                    Network_N_matrix[edges_[j][0]][edges_[j][1]]=bandwith_table[temp_H_list[r].edge_weight_choices[j]]
                    Network_N_matrix[edges_[j][1]][edges_[j][0]]=Network_N_matrix[edges_[j][0]][edges_[j][1]]
                
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 2 ) and (user_input[edges_[j][0]][edges_[j][1]] == 5):#RF/satellite:
        
                    Network_N_matrix[edges_[j][0]][edges_[j][1]]=bandwith_table[temp_H_list[r].edge_weight_choices[j] * 2]
                    Network_N_matrix[edges_[j][1]][edges_[j][0]]=Network_N_matrix[edges_[j][0]][edges_[j][1]]
                
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 2 ) and (user_input[edges_[j][0]][edges_[j][1]] == 6):#gray/satellite:
        
                    Network_N_matrix[edges_[j][0]][edges_[j][1]]=bandwith_table[temp_H_list[r].edge_weight_choices[j] + 1]
                    Network_N_matrix[edges_[j][1]][edges_[j][0]]=Network_N_matrix[edges_[j][0]][edges_[j][1]]
                
                
                #THREE EDGES
            
            
                elif (Connectivity_Matrix[edges_[j][0]][edges_[j][1]] == 3 ):#RF/gray/satellite:
        
                    Network_N_matrix[edges_[j][0]][edges_[j][1]]=bandwith_table[temp_H_list[r].edge_weight_choices[j]]
                    Network_N_matrix[edges_[j][1]][edges_[j][0]]=Network_N_matrix[edges_[j][0]][edges_[j][1]]
                
            
            
            
            
            
            
            #--------
            # HEFT calculation:
            #-----------------
            
            Performance=[]
        
        
            
            
            for t in range(0,len(nodes)): # loop for different task graphs of "n" nodes
                
        
                # Functions used by the HEFT script:  
                #----------------------------------
                
                def compcost(job, agent):
                    
                    comp_cost=DAG_matrix[job][job] / Network_N_matrix[agent][agent]
                    
                    return comp_cost
                
                
                def commcost(ni, nj, agent_A, agent_B):
                
                
                
                    if(agent_A == agent_B):
                        return 0
                    else:
                        
                        comm_cost = DAG_matrix[ni][nj] / Network_N_matrix[agent_A][agent_B] 
                        return comm_cost
                
                #---------  
                DAG_matrix=DAG_matrices[t]
                orders, jobson = cor.schedule(dags[t], H_list[r], compcost, commcost)
                #for eachP in sorted(orders):
                 #   print(eachP,orders[eachP])
                #print(jobson)
                
                
                # Performance and Cost Calculation: P(N,T) and C(N)
                #----------------------------------------------------
                
                #Performance:
                end_times=[]
                for i in range(0,len(orders)):
                    for j in range(0,len(orders[list(H_list[r])[i]])):
                        end_times.append(orders[list(H_list[r])[i]][j][2])
         
           
                
                Performance.append(max(end_times)) # total time to finish all the jobs
        
            Average_Performance.append(np.mean(Performance))
            #subgraph_nodes=list(H_list[r])
            #for i in range(0,lensub):
            temp_cost=0    
            for n in H_list[r]:
                temp_cost+=Cost_Matrix[n][n]
            Cost_vector.append(temp_cost)   
            
    return Average_Performance, Cost_vector
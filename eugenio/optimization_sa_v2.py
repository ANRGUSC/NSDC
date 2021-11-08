
"""

This script uses this HEFT script:
https://github.com/mackncheesiest/heft


"""
from heft import heft, gantt #new HEFT
import time

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
import random as ran

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
    
    
def optimization(temp_H_list,Connectivity_Matrix,Network_N_matrix,nodes,bandwith_table,user_input,Cost_Matrix,DAG_matrices,dags,G_dags):
    
    Average_Performance=[]
    Cost_vector=[]
    
    """
    number_of_subgraphs=[]
    temp_H_list=[]
    
    
    


    for obj in H_list_connectivities:
       
       number_of_subgraphs.append(obj.total_len())
       for i in range(0,obj.total_len()):
           temp_H_list.append(all_subgraphs(obj.N_subgraph, obj.edge_weight_choices[i]))
    """
    H_list=[]
    for obj in temp_H_list:
        H_list.append(obj.N_subgraph)
    

    #def_neighbour=ran.randrange(0,sum(number_of_subgraphs))
    
    
    for r in range(0,1): # different Mother Network's subgraphs loop
        
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
                
            
            
            
            
            
            
            #--------
            # HEFT calculation:
            #-----------------
            
            Performance=[]
        
        
            sparse_matrix=deepcopy(Network_N_matrixb)
            
            sparse_matrix=np.take(sparse_matrix, list(H_list[r].nodes), axis=0)
            sparse_matrix=np.take(sparse_matrix, list(H_list[r].nodes), axis=1)
            
            for t in range(0,len(nodes)): # loop for different task graphs of "n" nodes
                
        

                
                
                ############## NEW HEFT #################
                DAG_matrix=DAG_matrices[t]
                
                # Functions used by the HEFT script:  
                #----------------------------------
                #comp_matrix=np.zeros((len(DAG_matrix),mother_network_nodes[r]))
                comp_matrix=np.zeros((len(DAG_matrix),len(H_list[r].nodes)))
                
                #comm_matrix=deepcopy(Network_N_matrix[0:mother_network_nodes[r], 0:mother_network_nodes[r]])
                #comm_matrix=deepcopy(Network_N_matrixb[0:len(H_list[r].nodes), 0:len(H_list[r].nodes)])
                sparse_matrix=deepcopy(Network_N_matrixb)
                
                sparse_matrix=np.take(sparse_matrix, list(H_list[r].nodes), axis=0)
                sparse_matrix=np.take(sparse_matrix, list(H_list[r].nodes), axis=1)
                comm_matrix=deepcopy(sparse_matrix)
                
                #comm_matrix=np.ones((mother_network_nodes[r],mother_network_nodes[r]))
                
                
                for i in range(0,len(comm_matrix)):
                    comm_matrix[i][i]=0
                
                target_nodes=list(H_list[r].nodes)
                
                for i in range(0,len(comp_matrix)):
                    #for j in range(0,len(comm_matrix)):# we work with a complete graph. we fill up everything
                    for j in range(0,len(comm_matrix)):# we work with a complete graph. we fill up everything
                       
                        comp_matrix[i][j]=DAG_matrix[int(i)][int(i)] / Network_N_matrixb[target_nodes[j]][target_nodes[j]]
                      
                     

                

                #---------  
                #DAG_matrix=DAG_matrices[t]
                #start = time.process_time()
                
                #plt.figure(s)
                #nx.draw(G_dag)
                #NEW HEFT:
                #------------
                L=np.zeros(len(comm_matrix))
               
                ######################

                #####################
                
                sched, task_sched_table, _ = heft.schedule_dag(G_dags[r],communication_matrix=comm_matrix,computation_matrix=comp_matrix,communication_startup=L)
                maxtime=max([event.end for _, event in task_sched_table.items()])
                #task_sched_table.items()
                
                
                #########################################
                
                """
                
                
                
                def compcost(job, agent):
                    
                    comp_cost=DAG_matrix[int(job)][int(job)] / Network_N_matrixb[agent][agent]
                    
                    return comp_cost
                
                
                def commcost(ni, nj, agent_A, agent_B):
                
                
                
                    if(agent_A == agent_B):
                        return 0
                    else:
                        
                        comm_cost = DAG_matrix[int(ni)][int(nj)] / Network_N_matrixb[agent_A][agent_B] 
                        return comm_cost
                
                #---------  
                print("started")
                DAG_matrix=DAG_matrices[t]
                
                
                orders, jobson = cor.schedule(dags[t], H_list[r], compcost, commcost)
                #for eachP in sorted(orders):
                 #   print(eachP,orders[eachP])
                #print(jobson)
                print("finished")
                
                # Performance and Cost Calculation: P(N,T) and C(N)
                #----------------------------------------------------
                
                #Performance:
                end_times=[]
                for i in range(0,len(orders)):
                    for j in range(0,len(orders[list(H_list[r])[i]])):
                        end_times.append(orders[list(H_list[r])[i]][j][2])
         
           
                
                Performance.append(max(end_times)) # total time to finish all the jobs
            """
            #Average_Performance.append(np.mean(Performance))
            Average_Performance=maxtime
            
            #subgraph_nodes=list(H_list[r])
            #for i in range(0,lensub):
            temp_cost=0    
            for n in H_list[r]:
                temp_cost+=Cost_Matrix[n][n]
            Cost_vector.append(temp_cost)   
            
    #return Average_Performance, Cost_vector, temp_H_list, Network_N_matrixb
    return Average_Performance, Cost_vector, temp_H_list, Network_N_matrixb




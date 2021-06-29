# This version eliminates some randomness to help the understanding of the plots.
# So, we fix agent's speed, bandwith, etc. Also the cost matrix is fixed.

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


class all_subgraphs: # list with each subgraph of the mother network N and each possible 
               #connectivity choice of edges
                       
    def __init__(self, N_subgraph, edge_weight_choices): 
        self.N_subgraph = N_subgraph
        self.edge_weight_choices = edge_weight_choices
           
    def total_len(self):
        tot_len=len(self.edge_weight_choices)
        return tot_len
    


# TASK GRAPH Pre-established number of nodes and edges:
#-----------------------------------------------------

# This is DATA that the user provides:
#-------------------------
number_of_tasks=3
nodes=[] #  "n" nodes
edges=[]
for i in range(1,number_of_tasks+1):
    nodes.append(ran.randrange(3,10))# random # of nodes for a task
    edges.append(ran.randrange(5,15))# random # of edges for a task
#-------------------------



dags=[]
DAG_matrices=[]

for t in range(0,len(nodes)): # loop for different task graphs of "n" nodes
    
    

    #Creation of the DAG with the pre-establisehd number of nodes and edges    
    G_dag,dag=DAG_generator.random_dag(nodes[t], edges[t])
    #plt.figure(0)
    #nx.draw(G_dag)
    #--------   
    
    
    # Here we create a matrix that collects more info (time to finish eac job/amount 
    #of data to be transferred between jobs) from the user:
        
    DAG_matrix=np.zeros((len(dag),len(dag)))
    #we fill up the diagonal elements with the times(cycles) to finish each job
    for i in range(0,len(dag)):
        DAG_matrix[i][i]=2*i#ran.randrange(1,20)# time/cycles to finish each job
    for i in range(0,len(dag)):
            for j in range(0,len(dag[i])):
                #if len(dag[i])!=0:
                DAG_matrix[i][dag[i][j]]=8#ran.randrange(1,20) #amount of data transferred [data units]
                DAG_matrix[dag[i][j]][i]=DAG_matrix[i][dag[i][j]]# we make it symmmetric
    
    DAG_matrices.append(DAG_matrix)
    dags.append(dag)
    #---------




# Mother Network N Creation:
#-------------------
# This is DATA that the user provides:
#-------------------------
N=4 # main graph number of nodes/agents
min_N=3 # minimum number of nodes for future subgraphs
#-------------------------


G,H_list = Network_N.subgraphs(N, min_N) # we send N and ,min_N to the module Network_N.
#plt.figure(1)
#nx.draw(G)

# We define the agent's speed 
Network_N_matrix=np.zeros((N,N))

for i in range(0,N):
    Network_N_matrix[i][i]=ran.randrange(1,5)# agent's speed [cycles/sec]

"""
for i in range(0,N):
    for j in range(i,N):# we work with a complete graph. we fill up everything
        
        if i!=j:
            Network_N_matrix[i][j]=2*j#ran.randrange(1,5)# bandwith of the edges between agents [data/sec]
            Network_N_matrix[j][i]=Network_N_matrix[i][j]# we make it symmetric
"""

# We define a cost matrix that's a diagonal matrix by the time being
Cost_Matrix=np.zeros((N,N))
for i in range(0,N):
    Cost_Matrix[i][i]=2*i#ran.randrange(1,20)# agent's cost [$ dollars]

    
  
    
# ----------------
# Here we generate the different possible subgraphs of each Mother Network N
# taking in account 3 different connectivity types between nodes (RF, gray
# network and satellite). The Connectivity_Matrix is the matrix where the user
# specifies what type of edges (RF, gray, satellite) and how many of them we have 
# between the different nodes.




Connectivity_Matrix=np.zeros((N,N))
for i in range(0,N):
    for j in range(i,N):# we work with a complete graph. we fill up everything
        
        if i!=j:
            Connectivity_Matrix[i][j]=ran.randrange(1,4)# bandwith of the edges between agents [data/sec]
            Connectivity_Matrix[j][i]=Connectivity_Matrix[i][j]# we make it symmetric


# This DATA is input by the user:
# -------------------------------    
# BANDWITH TABLE FOR RF/GRAY/SATELLITE
bandwith_table=[2,4,6] #[data/sec]
user_input=np.zeros((N,N))
# USER INPUT Matrix PATTERN:
# (i,j)=1 --> RF edge
# (i,j)=2 --> gray edge
# (i,j)=3 --> satellite edge
# (i,j)=4 --> RF/gray edge
# (i,j)=5 --> RF/satellite edge
# (i,j)=6 --> gray/satellite edge
# (i,j)=7 --> RF/gray/satellite edge

for i in range(0,N):
    for j in range(i,N):# we fill up everything in the "user_input" matrix in 
                        # case we work with a complete graph.
        
        if i!=j:
            
            if Connectivity_Matrix[i][j]==1:
                
                user_input[i][j]=ran.randrange(1,4)# RF/gray/satellite
                user_input[j][i]=user_input[i][j]# we make it symmetric
            
            elif Connectivity_Matrix[i][j]==2:
                user_input[i][j]=ran.randrange(4,7)
                user_input[j][i]=user_input[i][j]
                
            elif Connectivity_Matrix[i][j]==3:    
            
                user_input[i][j]=7
                user_input[j][i]=user_input[i][j]

#---------------------------------------------
 
        
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
    #IGNORE THIS COMMENTS:comb=combinations(range(int(temp_sum)),len(edges_))
    
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
    
    #IGNORE THIS COMMENTS: Network_N_matrix[i][j]=2# bandwith of the edges between agents [data/sec]
    # depending on the edge connectivity type (RF, gray, satellite)
    #Network_N_matrix[j][i]=Network_N_matrix[i][j]# we make it symmetric



#----- MAIN LOOP ------
#----------------------


# ---------------- Here basically comes the two big "for" loops that calculates the average
# performance time/cost to perform/implement all the tasks in each subgraph (extracted 
#from the Mothe Network) --------------------------




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
    
    
#-------- Plots ------- :
#----------------------
plt.figure(1)
plt.plot(Cost_vector,Average_Performance,'bo',label='Performance')
#plt.plot(np.arange(len(H_list[r])),Average_Performance,label='Performance')
plt.legend()
#plt.plot(np.arange(len(H_list)),Cost,label='Cost')
#plt.legend()
plt.xlabel('Subgraph Cost')
plt.ylabel('Average Performance [time units]')
plt.grid(True)







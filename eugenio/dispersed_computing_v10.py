import numpy as np
import networkx as nx
import random as ran
import matplotlib.pyplot as plt
import heft
import core as cor
import Network_N
import DAG_generator




# TASK GRAPH Pre-established number of nodes and edges:
#-----------------------------------------------------
number_of_tasks=3
nodes=[]
edges=[]
for i in range(1,number_of_tasks+1):
    nodes.append(ran.randrange(3,10))# random # of nodes for a task
    edges.append(ran.randrange(5,15))# random # of nodes for a task


dags=[]
DAG_matrices=[]

for t in range(0,len(nodes)): # loop for different subgraphs of N nodes
    
    

    #Creation of the DAG with the pre-establisehd number of nodes and edges    
    G_dag,dag=DAG_generator.random_dag(nodes[t], edges[t])
    #plt.figure(0)
    #nx.draw(G_dag)
    #--------   

    DAG_matrix=np.zeros((len(dag),len(dag)))
    #we fill up the diagonal elements with the times(cycles) to finish each job
    for i in range(0,len(dag)):
        DAG_matrix[i][i]=ran.randrange(1,20)# time/cycles to finish each job
    for i in range(0,len(dag)):
            for j in range(0,len(dag[i])):
                #if len(dag[i])!=0:
                DAG_matrix[i][dag[i][j]]=ran.randrange(1,20) #amount of data transferred [data units]
                DAG_matrix[dag[i][j]][i]=DAG_matrix[i][dag[i][j]]# we make it symmmetric
    
    DAG_matrices.append(DAG_matrix)
    dags.append(dag)
    #---------



# Network N Creation:
#-------------------
N=5 # main graph number of nodes/agents
min_N=3 # minimum number of nodes for future subgraphs

G,H_list = Network_N.subgraphs(N, min_N) # we send N and ,min_N
#plt.figure(1)
#nx.draw(G)

# We define the agent's speed and the different bandwiths
Network_N_matrix=np.zeros((N,N))

for i in range(0,N):
    Network_N_matrix[i][i]=ran.randrange(1,5)# agent's speed [cycles/sec]

for i in range(0,N):
    for j in range(i,N):# we work with a complete graph. we fill up everything
        
        if i!=j:
            Network_N_matrix[i][j]=ran.randrange(1,5)# bandwith of the edges between agents [data/sec]
            Network_N_matrix[j][i]=Network_N_matrix[i][j]# we make it symmetric

# We define a cost matrix that's a diagonal matrix by the time being
Cost_Matrix=np.zeros((N,N))
for i in range(0,N):
    Cost_Matrix[i][i]=ran.randrange(1,20)# agent's cost [$ dollars]

    
  
 
   


Average_Performance=[]
Cost_vector=[]

for r in range(0,len(H_list)): # different Network's loop
    
 
    
 
    
    #--------
    # HEFT calculation:
    #-----------------
    
    Performance=[]


    
    
    for t in range(0,len(nodes)): # loop for different subgraphs of N nodes
        

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
    
#-------- Plots :
#---------------
plt.figure(r)
plt.plot(Cost_vector,Average_Performance,'bo',label='Performance')
#plt.plot(np.arange(len(H_list[r])),Average_Performance,label='Performance')
plt.legend()
#plt.plot(np.arange(len(H_list)),Cost,label='Cost')
#plt.legend()
plt.xlabel('Subgraph Cost')
plt.ylabel('Average Performance [time units]')
plt.grid(True)





from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import networkx as nx
import random as ran
import matplotlib.pyplot as plt
import heft
import core as cor
import Network_N
import DAG_generator
import time

# USER SET UP :
#-------------

number_of_tasks=np.arange(1,8,1) # number of task graphs

# mother network node sizes:
min_number_of_nodes=100
max_number_of_nodes=111

mother_network_nodes=np.arange(min_number_of_nodes,max_number_of_nodes,5)

Number_of_graphs=len(mother_network_nodes)

time_matrix=np.zeros((Number_of_graphs,len(number_of_tasks)))

for s in number_of_tasks:
    
    # TASK GRAPH Pre-established number of nodes and edges:
    #-----------------------------------------------------
    
    nodes=[]
    edges=[]
    #for i in range(1,number_of_tasks+1):
    for i in range(1,s+1):
        #nodes.append(ran.randrange(3,10))# random # of nodes for a task
        #edges.append(ran.randrange(5,15))# random # of edges for a task
        nodes.append(ran.randrange(3,4))# random # of nodes for a task
        edges.append(ran.randrange(3,4))# random # of edges for a task
    
    
    dags=[]
    DAG_matrices=[]
    
    #for t in range(0,number_of_tasks[s]): # creation of different task graphs
      
    for t in range(0,s): # creation of different task graphs
        
        
    
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
    H_list=[]
    
    
    for i in mother_network_nodes:
    
        N=i # main graph number of nodes/agents
        min_N=3 # minimum number of nodes for future subgraphs
        
        #G,H_list = Network_N.subgraphs(N, min_N) # we send N and ,min_N
        
        #G=nx.random_internet_as_graph(N)
        G = nx.complete_graph(N)
        #H_list=H_list.append(G)
        H_list.append(G)
    
        
        #plt.figure(1)
        #nx.draw(G)
    
    
    #Number_of_graphs=len(H_list)
    
    # We define the agent's speed and the different bandwiths
    
    Network_N_matrix=np.zeros((max_number_of_nodes,max_number_of_nodes))
    
    for i in range(0,max_number_of_nodes):
        Network_N_matrix[i][i]=ran.randrange(1,5)# agent's speed [cycles/sec]
    
    for i in range(0,max_number_of_nodes):
        for j in range(i,N):# we work with a complete graph. we fill up everything
            
            if i!=j:
                Network_N_matrix[i][j]=ran.randrange(1,5)# bandwith of the edges between agents [data/sec]
                Network_N_matrix[j][i]=Network_N_matrix[i][j]# we make it symmetric
    
    
    
    # We define a cost matrix that's a diagonal matrix by the time being
    Cost_Matrix=np.zeros((max_number_of_nodes,max_number_of_nodes))
    for i in range(0,max_number_of_nodes):
        Cost_Matrix[i][i]=ran.randrange(1,20)# agent's cost [$ dollars]
    
        
      
    
    
    
       
    
    
    Average_Performance=[]
    Cost_vector=[]
    
    
    
    
    # -----LOOPs------
    
    for r in range(0,Number_of_graphs): # different Network's loop
        
        #start = time.perf_counter()
        start = time.process_time()
        
     
        
        #--------
        # HEFT calculation:
        #-----------------
        
        Performance=[]
    
    
        
        #for t in range(0,number_of_tasks[s]): # loop for different subgraphs of N nodes
      
        for t in range(0,s): # loop for different subgraphs of N nodes
            
    
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
    
        #stop = time.perf_counter()
        stop = time.process_time()
        total_time=stop-start
        time_matrix[r][s-1]=total_time
        print(f"{total_time} seconds")
    
    



#-------- Plots :
#---------------

for i in range(0,len(time_matrix[0])):
    
    plt.figure(1)
    plt.plot(mother_network_nodes,time_matrix[:,i],label='Number of task graphs: %.2f' %number_of_tasks[i])
    #plt.plot(x, y, label='y = %.2f x + %.2f' %(A, B))
    #plt.plot(np.arange(len(H_list[r])),Average_Performance,label='Performance')
    plt.legend()
    #plt.plot(np.arange(len(H_list)),Cost,label='Cost')
    #plt.legend()
    plt.xlabel('Number of Network Nodes')
    plt.ylabel('Computation Time [seconds]')
    plt.grid(True)





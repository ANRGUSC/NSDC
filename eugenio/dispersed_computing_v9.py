import numpy as np
import networkx as nx
import random as ran
import matplotlib.pyplot as plt
import heft
import core as cor
import Network_N
import DAG_generator



# DAG data:
#---------
nodes=[10,9,8]
edges=[8,15,15]




for r in range(0,len(nodes)): # different DAG's loop
        
    G_dag,dag=DAG_generator.random_dag(nodes[r], edges[r])
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
    #---------
    
    
    
    # Network N data:
    #---------------
    
    N=5 # main graph number of nodes/agents
    min_N=3 # minimum number of nodes for the subgraph
    
    G,H_list = Network_N.subgraphs(N, min_N) # we send N and ,min_N
    #plt.figure(1)
    #nx.draw(G)
    
    #temp=len(H_list[0].nodes)
    Network_N_matrix=np.zeros((N,N))
    
    for i in range(0,N):
        Network_N_matrix[i][i]=ran.randrange(1,5)# agent's speed [cycles/sec]
    
    for i in range(0,N):
        for j in range(i,N):# we work with a complete graph. we fill up everything
            
            if i!=j:
                Network_N_matrix[i][j]=ran.randrange(1,5)# bandwith of the edges between agents [data/sec]
                Network_N_matrix[j][i]=Network_N_matrix[i][j]# we make it symmetric
    
    
    
    
    
    
    #--------
    # HEFT calculation:
    #-----------------
    
    Performance=[]
    Cost=[]
    
    
    for t in range(0,len(H_list)): # loop for different subgraphs of N nodes
        
        
        def compcost(job, agent):
            
            comp_cost=DAG_matrix[job][job] / Network_N_matrix[agent][agent]
            
            return comp_cost
        
        
        def commcost(ni, nj, agent_A, agent_B):
        
        
        
            if(agent_A == agent_B):
                return 0
            else:
                
                comm_cost = DAG_matrix[ni][nj] / Network_N_matrix[agent_A][agent_B] 
                return comm_cost
        
        
        orders, jobson = cor.schedule(dag, list(H_list[t]), compcost, commcost)
        for eachP in sorted(orders):
            print(eachP,orders[eachP])
        print(jobson)
        
        
        
        
        
        # Performance and Cost Calculation: P(N,T) and C(N,T)
        #----------------------------------------------------
        
        #Performance:
        end_times=[]
        for i in range(0,len(orders)):
            for j in range(0,len(orders[list(H_list[t])[i]])):
                end_times.append(orders[list(H_list[t])[i]][j][2])
                
        #Cost:
        start_minus_end_times=[]
        for i in range(0,len(orders)):
            for j in range(0,len(orders[list(H_list[t])[i]])):
                temp=orders[list(H_list[t])[i]][j][2]-orders[list(H_list[t])[i]][j][1]
                start_minus_end_times.append(temp)    
        
        Performance.append(max(end_times)) # total time to finish all the jobs
        Cost.append(sum(start_minus_end_times))# sum of all the individual times to finish each job
    
    
    #-------- Plots :
    #---------------
    plt.figure(r)
    plt.plot(np.arange(len(H_list)),Performance,label='Performance')
    plt.legend()
    plt.plot(np.arange(len(H_list)),Cost,label='Cost')
    plt.legend()
    plt.xlabel('Subgraph #')
    plt.ylabel('[time units]')
    plt.grid(True)




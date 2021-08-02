
import numpy as np
import random as ran
import matplotlib.pyplot as plt
import DAG_generator
import all_mother_network_subgraphs
import optimization
from simulated_annealing import sa
import sa_v1

class all_subgraphs: # list with each subgraph of the mother network N and each possible 
                     # connectivity choice of edges
                       
    def __init__(self, N_subgraph, edge_weight_choices): 
        self.N_subgraph = N_subgraph
        self.edge_weight_choices = edge_weight_choices
           
    def total_len(self):
        tot_len=len(self.edge_weight_choices)
        return tot_len
    



#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

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
    
    
    # Here we create a matrix that collects more info (time to finish each job and amount 
    # of data to be transferred between jobs) from the user:
        
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

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------






#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# Mother Network N Creation:
#-------------------
# This is DATA that the user provides:
#-------------------------
N=4 # main graph number of nodes/agents
min_N=3 # minimum number of nodes for future subgraphs
#-------------------------


# This is more DATA that eventually might be input by the user:
#-------------
# We define the agent's speed (node's speed)
Network_N_matrix=np.zeros((N,N))

for i in range(0,N):
    Network_N_matrix[i][i]=ran.randrange(1,5)# agent's speed [cycles/sec]


# We define a cost matrix that's a diagonal matrix by the time being
Cost_Matrix=np.zeros((N,N))
for i in range(0,N):
    Cost_Matrix[i][i]=2*i#ran.randrange(1,20)# agent's cost [$ dollars]
#---------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------






#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
    
# ----------------
# Here we generate the different possible subgraphs of the Mother Network N
# taking in account 3 different connectivity types between nodes (RF, gray
# network and satellite). The Connectivity_Matrix is the matrix where the user
# specifies what type of edges (RF, gray, satellite) and how many of them we have 
# between the different nodes.




Connectivity_Matrix=np.zeros((N,N))
for i in range(0,N):
    for j in range(i,N):# we work with a complete graph. we fill up everything
        
        if i!=j:
            Connectivity_Matrix[i][j]=ran.randrange(1,4)#number of edges(RF/gray/Satellite)
                                                        #per pair of nodes
            Connectivity_Matrix[j][i]=Connectivity_Matrix[i][j]# we make it symmetric


# This DATA is input by the user:
# -------------------------------    
# BANDWITH TABLE FOR RF/GRAY/SATELLITE
bandwith_table=[2,4,6] #[data/sec]

# The "user_input" matrix contains the specific amount of edges per pair of nodes:
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

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------




#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

#----- MAIN LOOP ------
#----------------------


# ---------------- Here basically comes the two big "for" loops that calculates the average
# performance time/cost to compute all the tasks in each Mother Network subgraph ---------


G,H_list_connectivities=all_mother_network_subgraphs.all_mother_network_subgraphs(N, min_N, Connectivity_Matrix)


Average_Performance,Cost_vector,all_H_list=optimization.optimization(H_list_connectivities,Connectivity_Matrix,Network_N_matrix,nodes,bandwith_table,user_input,Cost_Matrix,DAG_matrices,dags)


#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

    

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





#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#---------------------
# Simulated Annealing:
#---------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------


f=len(all_H_list)
bounds=[1,f-1]
x0=ran.randrange(f)

opt = sa_v1.minimize(Average_Performance, x0,bounds, opt_mode='continuous', step_max=1000, t_max=1, t_min=0)

opt.results()
opt.best_state


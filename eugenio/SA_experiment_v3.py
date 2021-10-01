#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import numpy as np
import random as ran
import matplotlib.pyplot as plt
import DAG_generator
import all_mother_network_subgraphs
import optimization_sa_v1
import optimization
#from simulated_annealing import sa
import sa_v2
import search_space
import networkx as nx
import DAG_generator_scaled_v2
import optimization_sa_multiple_subgraphs_v1

import all_mother_network_subgraphs_sa_v2

class all_subgraphs: # list with each subgraph of the mother network N and each possible 
                     # connectivity choice of edges
                       
    def __init__(self, N_subgraph, edge_weight_choices): 
        self.N_subgraph = N_subgraph
        self.edge_weight_choices = edge_weight_choices
           
    def total_len(self):
        tot_len=len(self.edge_weight_choices)
        return tot_len
    
class all_data:
    
    def __init__(self, G,min_N,N,Connectivity_Matrix,Network_N_matrix,number_of_tasks,bandwith_table,user_input,Cost_Matrix,DAG_matrices,dags): 
        
               
        self.G = G 
        self.min_N = min_N
        self.N = N 
        self.Connectivity_Matrix = Connectivity_Matrix
        self.Network_N_matrix = Network_N_matrix
        self.number_of_tasks = number_of_tasks
        self.bandwith_table = bandwith_table
        self.user_input = user_input
        self.Cost_Matrix = Cost_Matrix 
        self.DAG_matrices = DAG_matrices
        self.dags = dags


# USER SET UP :
#-------------

number_of_tasks=np.arange(4,6,2) # number of task graphs

# mother network node sizes:
min_number_of_nodes=10
max_number_of_nodes=61
step=10

mother_network_nodes=np.arange(min_number_of_nodes,max_number_of_nodes,step)

Number_of_graphs=len(mother_network_nodes)

time_matrix=np.zeros((Number_of_graphs,len(number_of_tasks)))


task_count=0

for s in number_of_tasks:
    
            
    
    task_count +=1
    
    # TASK GRAPH Pre-established number of nodes and edges:
    #-----------------------------------------------------
    
    nodes=[]
    edges=[]
    #for i in range(1,number_of_tasks+1):
    for i in range(1,s+1):
        #Note: the number of edges can not be lower than (nodes-1)
        nodes.append(s)# random # of nodes for a task
        edges.append(ran.randrange(5,15))# random # of edges for a task
        
        #nodes.append(ran.randrange(3,8))# random # of nodes for a task
        #edges.append(ran.randrange(3,10))# random # of edges for a task
    
    
    dags=[]
    DAG_matrices=[]
    
    #for t in range(0,number_of_tasks[s]): # creation of different task graphs
      
    for t in range(0,1): # creation of a task graph
        
        
    
        #Creation of the DAG that scales either as O(n) or O(n^2), with the pre-establisehd 
        #number of nodes.  
        G_dag,dag=DAG_generator_scaled_v2.random_dag(nodes[t],'n')
        #plt.figure(0)
        #nx.draw(G_dag)
        #--------   
    
        DAG_matrix=np.zeros((len(dag),len(dag)))
        #we fill up the diagonal elements with the times(cycles) to finish each job
        for i in range(0,len(dag)):
            DAG_matrix[i][i]=ran.randrange(1,5)# time/cycles to finish each job
        for i in range(0,len(dag)):
                for j in range(0,len(dag[i])):
                    #if len(dag[i])!=0:
                    DAG_matrix[i][int(dag[i][j])]=ran.randrange(1,5) #amount of data transferred [data units]
                    DAG_matrix[int(dag[i][j])][i]=DAG_matrix[i][int(dag[i][j])]# we make it symmmetric
        
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


"""

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

#----- MAIN LOOP ------
#----------------------


# ---------------- Here basically comes the two big "for" loops that calculates the average
# performance time/cost to compute all the tasks in each Mother Network subgraph ---------


G,H_list_connectivities=all_mother_network_subgraphs.all_mother_network_subgraphs(N, min_N, Connectivity_Matrix)


search_space_results=search_space.search_space(H_list_connectivities,Connectivity_Matrix,Network_N_matrix,nodes,bandwith_table,user_input)


Average_Performance,Cost_vector,all_H_list=optimization_sa_multiple_subgraphs_v1.optimization(H_list_connectivities,Connectivity_Matrix,Network_N_matrix,number_of_tasks,bandwith_table,user_input,Cost_Matrix,DAG_matrices,dags)


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

sa_vector=np.zeros((f,1))
sa_vector[opt.best_state][0]=1

#-------- Plots ------- :
#----------------------
plt.figure(2)
plt.plot(Average_Performance)
plt.plot(sa_vector)
#plt.plot(np.arange(len(H_list[r])),Average_Performance,label='Performance')
plt.legend()
#plt.plot(np.arange(len(H_list)),Cost,label='Cost')
#plt.legend()
plt.xlabel('Subgraph')
plt.ylabel('Average Performance [time units]')
plt.grid(True)

"""

G = nx.complete_graph(N)

# Neighbour Function:
#-------------------
neighbour_number_of_nodes=ran.randrange(min_N,N)
H_list_connectivities=all_mother_network_subgraphs_sa_v2.all_mother_network_subgraphs(G,N,neighbour_number_of_nodes, Connectivity_Matrix)

if len(H_list_connectivities[0].edge_weight_choices) != 1:
    tt=ran.randrange(0,len(H_list_connectivities[0].edge_weight_choices)-1)
else:
    tt=ran.randrange(0,len(H_list_connectivities[0].edge_weight_choices))
    
    
temp_H_list=[]
temp_H_list.append(all_subgraphs(H_list_connectivities[0].N_subgraph, H_list_connectivities[0].edge_weight_choices[tt]))

#def_neighbour=ran.randrange(0,H_list_connectivities.total_len())




# Cost Function (HEFT by the time being):
#-------------------------------------
Average_Performance,Cost_vector,picked_H_list,picked_Network_N_matrixb=optimization_sa_v1.optimization(temp_H_list,Connectivity_Matrix,Network_N_matrix,number_of_tasks,bandwith_table,user_input,Cost_Matrix,DAG_matrices,dags)
print(f"{Average_Performance}")

       
        
info=all_data(G,min_N,N,Connectivity_Matrix,Network_N_matrix,number_of_tasks,bandwith_table,user_input,Cost_Matrix,DAG_matrices,dags)
#f=len(all_H_list)
bounds=[0,1]
#x0=ran.randrange(f)
x0=temp_H_list
opt = sa_v2.minimize(info,optimization_sa_v1, x0,bounds, opt_mode='continuous', step_max=1000, t_max=1, t_min=0)

opt.results()
opt.best_state

#plt.figure(1)
#uu=opt.best_state[0].N_subgraph
#nx.draw(G)





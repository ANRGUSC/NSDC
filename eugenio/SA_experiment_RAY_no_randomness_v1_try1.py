#from IPython import get_ipython
#get_ipython().magic('reset -sf')



###########
#We use the add/delete SA neighbour function (sa_v5) and we use 6 cores that 
#start the analysis with different X0 neighbor value !!!
###########


import numpy as np
import random as ran
import matplotlib.pyplot as plt
#import DAG_generator
#import all_mother_network_subgraphs
import optimization_sa_v2
#import optimization
#from simulated_annealing import sa
import sa_v6b_ray_v1
import sa_v5
#import search_space
import networkx as nx
import DAG_generator_scaled_v6_one_sink
#import optimization_sa_multiple_subgraphs_v1
import time
import all_mother_network_subgraphs_sa_v2
from itertools import combinations
import dill                            #pip install dill --user

import ray
import time


#start = time.process_time()
start = time.time()

ray_tasks=6
#SA_steps=50

ray.shutdown()
# Start Ray.
ray.init()

@ray.remote
def f(info,optimization_sa_v2, x0,bounds,SA_steps, opt_mode, t_max, t_min):
    
    
    #opt = sa_v6b_ray_v1.minimize(info,optimization_sa_v2, x0,bounds, opt_mode, partition_trimmed, step_max=SA_steps, t_max=1, t_min=0)
    
    opt = sa_v5.minimize(info,optimization_sa_v2, x0,bounds, opt_mode, step_max=SA_steps, t_max=1, t_min=0)

    return opt



class all_subgraphs: # list with each subgraph of the mother network N and each possible 
                     # connectivity choice of edges
                       
    def __init__(self, N_subgraph, edge_weight_choices): 
        self.N_subgraph = N_subgraph
        self.edge_weight_choices = edge_weight_choices
           
    def total_len(self):
        tot_len=len(self.edge_weight_choices)
        return tot_len
    
class all_data:
    
    def __init__(self, G,min_N,N,Connectivity_Matrix,Network_N_matrix,number_of_tasks,bandwith_table,user_input,Cost_Matrix,DAG_matrices,dags,G_dags): 
        
               
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
        self.G_dags = G_dags



#filename = '10_to_60.pkl'
#dill.load_session(filename)

#start = time.process_time()
#start = time.time()

# USER SET UP :
#-------------
print("SA begins")
number_of_tasks=np.arange(200,202,3) # number of task graphs / minimum is 6 !!

# mother network node sizes:
min_number_of_nodes=150
max_number_of_nodes=159
step=500

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
    G_dags=[]
    DAG_matrices=[]
    
    #for t in range(0,number_of_tasks[s]): # creation of different task graphs
      
    for t in range(0,1): # creation of a task graph
        
        
    
        #Creation of the DAG that scales either as O(n) or O(n^2), with the pre-establisehd 
        #number of nodes.  
        G_dag,dag=DAG_generator_scaled_v6_one_sink.random_dag(nodes[t],'n')
        """
        plt.figure(0)
        nx.draw(G_dag)
        """
        #--------   
    
        """
        DAG_matrix=np.zeros((len(dag),len(dag)))
        #we fill up the diagonal elements with the times(cycles) to finish each job
        for i in range(0,len(dag)):
            DAG_matrix[i][i]=ran.randrange(1,5)# time/cycles to finish each job
        for i in range(0,len(dag)):
                for j in range(0,len(dag[i])):
                    #if len(dag[i])!=0:
                    DAG_matrix[i][int(dag[i][j])]=ran.randrange(1,5) #amount of data transferred [data units]
                    DAG_matrix[int(dag[i][j])][i]=DAG_matrix[i][int(dag[i][j])]# we make it symmmetric
                    # We add the corresponding weigths to the DAG graph:
                    G_dag[i][int(dag[i][j])]["weight"]=DAG_matrix[i][int(dag[i][j])]
                    
        DAG_matrices.append(DAG_matrix)
        dags.append(dag)
        G_dags.append(G_dag)
        #---------
        """
        #####added
        DAG_matrix=np.zeros((len(dag),len(dag)))
        #we fill up the diagonal elements with the times(cycles) to finish each job
        for i in range(0,len(dag)-1):#the last node is the sink; thus, we put "0" because there is no task here
            DAG_matrix[i][i]=4#ran.randrange(1,5)# time/cycles to finish each job
            
        
        for i in range(0,len(dag)):
            DAG_matrix[i][i]=5#ran.randrange(1,5)# time/cycles to finish each job
        for i in range(0,len(dag)):
                for j in range(0,len(dag[i])):
                    #if len(dag[i])!=0:
                        
                    #DAG_matrix[i][int(dag[i][j])]=ran.randrange(1,5) #amount of data transferred [data units]
                    
                    # We add the corresponding weigths to the DAG matrix:
                    DAG_matrix[i][int(dag[i][j])]=G_dag[i][int(dag[i][j])]["weight"]
                    DAG_matrix[int(dag[i][j])][i]=DAG_matrix[i][int(dag[i][j])]# we make it symmmetric


                    
        DAG_matrices.append(DAG_matrix)
        dags.append(dag)
        G_dags.append(G_dag)
        #####

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------






#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
stored_min_energy=[]
N_nodes=range(min_number_of_nodes,max_number_of_nodes,step)



#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# HERE WE TRY TO MAKE EACH MOTHER NETWORK A SUBGRAPH OF A BIG MAIN 
# MOTHER NETWORK.


# Mother Network N Creation:
#-------------------
# This is DATA that the user provides:
#-------------------------
N=max_number_of_nodes # main graph number of nodes/agents
min_N=3 # minimum number of nodes for future subgraphs
#-------------------------


# This is more DATA that eventually might be input by the user:
#-------------
# We define the agent's speed (node's speed)
Network_N_matrixa=np.zeros((N,N))

for i in range(0,N):
    Network_N_matrixa[i][i]=4#ran.randrange(1,5)# agent's speed [cycles/sec]


# We define a cost matrix that's a diagonal matrix by the time being
Cost_Matrixa=np.zeros((N,N))
for i in range(0,N):
    Cost_Matrixa[i][i]=2*i#ran.randrange(1,20)# agent's cost [$ dollars]

    
# ----------------
# Here we generate the different possible subgraphs of the Mother Network N
# taking in account 3 different connectivity types between nodes (RF, gray
# network and satellite). The Connectivity_Matrix is the matrix where the user
# specifies what type of edges (RF, gray, satellite) and how many of them we have 
# between the different nodes.




Connectivity_Matrixa=np.zeros((N,N))
for i in range(0,N):
    for j in range(i,N):# we work with a complete graph. we fill up everything
        
        if i!=j:
            Connectivity_Matrixa[i][j]=3#ran.randrange(1,4)#number of edges(RF/gray/Satellite)
                                                        #per pair of nodes
            Connectivity_Matrixa[j][i]=Connectivity_Matrixa[i][j]# we make it symmetric


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
            
            if Connectivity_Matrixa[i][j]==1:
                
                user_input[i][j]=ran.randrange(1,4)# RF/gray/satellite
                user_input[j][i]=user_input[i][j]# we make it symmetric
            
            elif Connectivity_Matrixa[i][j]==2:
                user_input[i][j]=ran.randrange(4,7)
                user_input[j][i]=user_input[i][j]
                
            elif Connectivity_Matrixa[i][j]==3:    
            
                user_input[i][j]=7
                user_input[j][i]=user_input[i][j]







"""
check=False
while check == False:
    
    #G = nx.erdos_renyi_graph(N, 0.5, directed=False)
    G=nx.hypercube_graph(N)
    #G=nx.dense_gnm_random_graph(N, N+2, seed=None)
    check=nx.is_connected(G)
"""


Ga = nx.complete_graph(N) 

"""  
plt.figure(1)
nx.draw(Ga)
plt.title('Mother Network')
"""

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------




for k in N_nodes:
    # Mother Network N Creation:
    #-------------------
    # This is DATA that the user provides:
    #-------------------------
    N=k # main graph number of nodes/agents
    min_N=3 # minimum number of nodes for future subgraphs
    #-------------------------
    
    
    # This is more DATA that eventually might be input by the user:
    #-------------
    # We define the agent's speed (node's speed)
    Network_N_matrix=Network_N_matrixa[0:N,0:N]
    
    
    
      
    # We define a cost matrix that's a diagonal matrix by the time being
    Cost_Matrix=Cost_Matrixa[0:N,0:N]

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
    
    
    
    
    Connectivity_Matrix=Connectivity_Matrixa[0:N,0:N]
    
    
 
    
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    
    
    # SIMULATED ANNEALING: -------
    #-----------------------------
    
    """
    check=False
    while check == False:
        
        #G = nx.erdos_renyi_graph(N, 0.5, directed=False)
        G=nx.hypercube_graph(N)
        #G=nx.dense_gnm_random_graph(N, N+2, seed=None)
        check=nx.is_connected(G)
    """
    
    sub_graphs_nodes=range(0,N)
    G = Ga.subgraph(sub_graphs_nodes) 
    
    """   
    plt.figure(2)
    nx.draw(G)
    plt.title('Mother Network')
    """
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    
    # Start 4 tasks in parallel.
    result_ids = []
    SA_steps=50
    for i in range(ray_tasks): 
    
        # Neighbour Function - Random initial subgraph choice:
        #----------------------------------------------------
        """
        neighbour_number_of_nodes=ran.randrange(min_N,N)
    
        
        comb = list(combinations(range(0,N),neighbour_number_of_nodes))
        i=ran.randrange(0,len(comb))
        initial_neighbour_nodes=comb[i]
        """
        
        ##########
        initial_neighbour_nodes = ran.sample(range(min_N, N), ran.randrange(min_N,len(range(min_N, N))))
        ##########
        
        H_list=[]
        H_list.append(G.subgraph(initial_neighbour_nodes))
        edges_=list(H_list[0].edges)
        edges_choice=[]
        for j in range(0,len(edges_)):
            edges_choice.append(ran.randrange(0,int(Connectivity_Matrix[edges_[j][0]][edges_[j][1]])))
        
        H_list_connectivities=[]
        H_list_connectivities.append(all_subgraphs(H_list[0],edges_choice))
        
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        
        
        
        temp_H_list = H_list_connectivities    
                
        info=all_data(G,min_N,N,Connectivity_Matrix,Network_N_matrix,number_of_tasks,bandwith_table,user_input,Cost_Matrix,DAG_matrices,dags,G_dags)
        #f=len(all_H_list)
        bounds=[0,1]
        #x0=ran.randrange(f)
        x0=temp_H_list
    
    #########################
    ##### Remote Tasks ########


        #partition_trimmed=np.trim_zeros(partition[i][:])
        
        result_ids.append(f.remote(info,optimization_sa_v2, x0,bounds, SA_steps, opt_mode='continuous',  t_max=1, t_min=0))
        #SA_steps+=100
        #opt = sa_v6b_ray_v1.minimize(info,optimization_sa_v2, x0,bounds, partition_trimmed, opt_mode='continuous', step_max=50, t_max=1, t_min=0)
    
    #########################
    #########################
    """
    opt = sa_v6b_ray_v0.minimize(info,optimization_sa_v2, x0,bounds, opt_mode='continuous', step_max=50, t_max=1, t_min=0)
    """
    
    # Wait for the tasks to complete and retrieve the results.
    # With at least 4 cores, this will take 1 second.
    results = ray.get(result_ids)  # [0, 1, 2, 3]

    best_subgraph=[]
    #opt.results()
    for i in range(ray_tasks):
        
        #[row[2] for row in results[i].hist]
        #stored_min_energy.append(min([row[2] for row in results[i].hist]))
        stored_min_energy.append(results[i].best_energy)
        best_subgraph.append(results[i].best_state)
        
#stop = time.process_time()
stop = time.time()

total_time=(stop-start)/3600
    
"""
plt.figure(3)
nx.draw(G)
plt.title('Mother Network')
"""

"""
plt.figure(2)
plt.plot([row[0] for row in opt.hist],[row[2] for row in opt.hist],'bo')
plt.xlabel('steps')
plt.ylabel('Makespan ("Energy")[time units]')
plt.grid(True)
plt.title('Makespan vs time steps')

plt.figure(3)
nx.draw(opt.best_state[0].N_subgraph)
plt.title('Best Subgraph')
"""

"""
plt.figure(6)
plt.plot(N_nodes,stored_min_energy,'bo')
plt.xlabel('steps')
plt.ylabel('Makespan ("Energy")[time units]')
plt.grid(True)
plt.title('Makespan vs Mother Network # of nodes')
"""

#print(results[0].best_energy)
#print(results[1].best_energy)


filename = '200DAGs_150Net_50SA_steps_v5_ray_v2_6cores_try1.pkl'
dill.dump_session(filename)
print("SA end")


#exit()
# and to load the session again:
#dill.load_session(filename)
























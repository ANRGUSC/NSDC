# addapted from https://github.com/nathanrooy/simulated-annealing

from random import randint
from random import random
from random import choice
from random import sample
from math import exp
from math import log
from random import randrange
import all_mother_network_subgraphs_sa_v2
import all_mother_network_subgraphs
from itertools import combinations


#--- MAIN ---------------------------------------------------------------------+
class all_subgraphs: # list with each subgraph of the mother network N and each possible 
                     # connectivity choice of edges
                       
    def __init__(self, N_subgraph, edge_weight_choices): 
        self.N_subgraph = N_subgraph
        self.edge_weight_choices = edge_weight_choices
           
    def total_len(self):
        tot_len=len(self.edge_weight_choices)
        return tot_len
    

class minimize():
    '''Simple Simulated Annealing
    '''

    def __init__(self, info,optimization_sa_v2, x0, bounds, opt_mode, cooling_schedule='linear', step_max=1000, t_min=0, t_max=100, alpha=None, damping=1):

        # checks
        assert opt_mode in ['combinatorial','continuous'], 'opt_mode must be either "combinatorial" or "continuous"'
        assert cooling_schedule in ['linear','exponential','logarithmic', 'quadratic'], 'cooling_schedule must be either "linear", "exponential", "logarithmic", or "quadratic"'


        # initialize starting conditions
        self.t = t_max
        self.t_max = t_max
        self.t_min = t_min
        self.step_max = step_max
        self.opt_mode = opt_mode
        self.hist = []
        self.cooling_schedule = cooling_schedule
        
        self.info = info
        self.cost_func = optimization_sa_v2
        self.x0 = x0
        self.bounds = bounds[:]
        self.damping = damping
        self.current_state = self.x0
        #self.current_energy = self.cost_func[self.x0]
        
        temp = self.cost_func.optimization(self.x0,self.info.Connectivity_Matrix,self.info.Network_N_matrix,self.info.number_of_tasks,self.info.bandwith_table,self.info.user_input,self.info.Cost_Matrix,self.info.DAG_matrices,self.info.dags,self.info.G_dags)
        self.current_energy = temp[0]
        
        self.best_state = self.current_state
        self.best_energy = self.current_energy


        # initialize optimization scheme
        if self.opt_mode == 'combinatorial': self.get_neighbor = self.move_combinatorial
        if self.opt_mode == 'continuous': self.get_neighbor = self.move_continuous


        # initialize cooling schedule
        if self.cooling_schedule == 'linear':
            if alpha != None:
                self.update_t = self.cooling_linear_m
                self.cooling_schedule = 'linear multiplicative cooling'
                self.alpha = alpha

            if alpha == None:
                self.update_t = self.cooling_linear_a
                self.cooling_schedule = 'linear additive cooling'

        if self.cooling_schedule == 'quadratic':
            if alpha != None:
                self.update_t = self.cooling_quadratic_m
                self.cooling_schedule = 'quadratic multiplicative cooling'
                self.alpha = alpha

            if alpha == None:
                self.update_t = self.cooling_quadratic_a
                self.cooling_schedule = 'quadratic additive cooling'

        if self.cooling_schedule == 'exponential':
            if alpha == None: self.alpha =  0.8
            else: self.alpha = alpha
            self.update_t = self.cooling_exponential_m

        if self.cooling_schedule == 'logarithmic':
            if alpha == None: self.alpha =  0.8
            else: self.alpha = alpha
            self.update_t = self.cooling_logarithmic_m


        # begin optimizing
        print("SA beginning")
        self.step, self.accept = 1, 0
        while self.step < self.step_max and self.t >= self.t_min and self.t>0:

            # get neighbor
            proposed_neighbor = self.get_neighbor()

            # check energy level of neighbor
            
            #E_n = self.cost_func[proposed_neighbor]
            #dE = E_n - self.current_energy
            temp_energy=self.cost_func.optimization(proposed_neighbor,self.info.Connectivity_Matrix,self.info.Network_N_matrix,self.info.number_of_tasks,self.info.bandwith_table,self.info.user_input,self.info.Cost_Matrix,self.info.DAG_matrices,self.info.dags,self.info.G_dags)
            E_n = temp_energy[0]
            dE = E_n - self.current_energy
            
            
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        
            # determine if we should accept the current neighbor
            
            # ( Here we always accept te proposed neighbor )
                
            #if random() < self.safe_exp(-dE / self.t):
            self.current_energy = E_n
            self.current_state = proposed_neighbor
            self.accept += 1
                
        
            # check if the current neighbor is best solution so far
            if E_n < self.best_energy:
                self.best_energy = E_n
                self.best_state = proposed_neighbor

            # persist some info for later
            self.hist.append([
                self.step,
                self.t,
                self.current_energy,
                self.best_energy])

            # update some stuff
            self.t = self.update_t(self.step)
            self.step += 1

        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        print("SA end")
        


        # generate some final stats
        self.acceptance_rate = self.accept / self.step


    def move_continuous(self):
        # preturb current state by a random amount
        #neighbor = self.current_state + (pow(-1,randrange(1,3)) * self.damping)
        
        """
        # Neighbour Function (0):
        #-------------------
        neighbor=randrange(self.info.min_N,self.info.N)
        H_list_connectivities=all_mother_network_subgraphs_sa_v2.all_mother_network_subgraphs(self.info.G,self.info.N,neighbor, self.info.Connectivity_Matrix)
        tt=randrange(0,len(H_list_connectivities[0].edge_weight_choices)-1)
        temp_H_list=[]
        temp_H_list.append(all_subgraphs(H_list_connectivities[0].N_subgraph, H_list_connectivities[0].edge_weight_choices[tt]))
        neighbor=temp_H_list
        """
        
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        
        """
        # Neighbour Function (1):
        #------------------------
        neighbour_number_of_nodes=randrange(self.info.min_N,self.info.N)
        comb = list(combinations(range(0,self.info.N),neighbour_number_of_nodes))
        i=randrange(0,len(comb))
        
        initial_neighbour_nodes=comb[i]
        H_list=[]
        H_list.append(self.info.G.subgraph(initial_neighbour_nodes))
        edges_=list(H_list[0].edges)
        edges_choice=[]
        for j in range(0,len(edges_)):
            edges_choice.append(randrange(0,int(self.info.Connectivity_Matrix[edges_[j][0]][edges_[j][1]])))
        
        H_list_connectivities=[]
        H_list_connectivities.append(all_subgraphs(H_list[0],edges_choice))
        neighbor = H_list_connectivities
        """
        
        
        """
        # Neighbour Function (2): This function adds/deletes a node to actual neighbor
        #-----------------------------------------------------------------------------
        
        # do we add a node or we delete one ?
        
        add_delete_node=randrange(0,2) #if equal to 0 we delete, if equal to 1: we add node
        actual_nodes=list(self.x0[0].N_subgraph.nodes)
        #add_delete_node=1
        if add_delete_node == 0:
            random_item_from_list = choice(actual_nodes)
            actual_nodes.remove(random_item_from_list)
        else:
            
            looking_for_nodes=[]
            for i in range(0,len(actual_nodes)):
                looking_for_nodes.extend(list(self.info.G.neighbors(actual_nodes[i])))
            
            looking_for_nodes = list(dict.fromkeys(looking_for_nodes))  
            #we delete actual nodes from the looking_for nodes list:
            for i in range(0,len(actual_nodes)):    
                 looking_for_nodes = [ x for x in looking_for_nodes if x!=actual_nodes[i] ]
                
            if looking_for_nodes == []:
                actual_nodes=actual_nodes
            else:
                new_node=choice(looking_for_nodes)
                actual_nodes.append(new_node)
            
        #---------------
        
        #neighbour_number_of_nodes=randrange(self.info.min_N,self.info.N)
        #comb = list(combinations(range(0,self.info.N),neighbour_number_of_nodes))
        #i=randrange(0,len(comb))
        initial_neighbour_nodes=actual_nodes
        
        H_list=[]
        H_list.append(self.info.G.subgraph(initial_neighbour_nodes))
        edges_=list(H_list[0].edges)
        edges_choice=[]
        for j in range(0,len(edges_)):
            edges_choice.append(randrange(0,int(self.info.Connectivity_Matrix[edges_[j][0]][edges_[j][1]])))
        
        H_list_connectivities=[]
        H_list_connectivities.append(all_subgraphs(H_list[0],edges_choice))
        neighbor = H_list_connectivities
        """
        
        
        # Neighbour Function (3): This function proposes a random neighbor
        #-----------------------------------------------------------------------------
        
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        
        ##########
        #initial_neighbour_nodes = sample(range(self.info.min_N, self.info.N), randrange(self.info.min_N,len(range(self.info.min_N, self.info.N))))
                
        initial_neighbour_nodes = sample(range(0, self.info.N), randrange(self.info.min_N,len(range(self.info.min_N, self.info.N))))
        
        ##########
        
        H_list=[]
        H_list.append(self.info.G.subgraph(initial_neighbour_nodes))
        edges_=list(H_list[0].edges)
        edges_choice=[]
        for j in range(0,len(edges_)):
            edges_choice.append(randrange(0,int(self.info.Connectivity_Matrix[edges_[j][0]][edges_[j][1]])))
        
        H_list_connectivities=[]
        H_list_connectivities.append(all_subgraphs(H_list[0],edges_choice))
        
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        
        
        
        neighbor = H_list_connectivities    
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------

        
        
        """
        # clip to upper and lower bounds
        if self.bounds:
          
                x_min, x_max = self.bounds
                neighbor = min(max(neighbor, x_min), x_max)
        """
        return neighbor


    def move_combinatorial(self):
        '''Swaps two random nodes along path
        Not the most efficient, but it does the job...
        '''
        p0 = randint(0, len(self.current_state)-1)
        p1 = randint(0, len(self.current_state)-1)

        neighbor = self.current_state[:]
        neighbor[p0], neighbor[p1] = neighbor[p1], neighbor[p0]

        return neighbor


    def results(self):
        print('+------------------------ RESULTS -------------------------+\n')
        print(f'      opt.mode: {self.opt_mode}')
        print(f'cooling sched.: {self.cooling_schedule}')
        if self.damping != 1: print(f'       damping: {self.damping}\n')
        else: print('\n')

        print(f'  initial temp: {self.t_max}')
        print(f'    final temp: {self.t:0.6f}')
        print(f'     max steps: {self.step_max}')
        print(f'    final step: {self.step}\n')

        #print(f'  final energy: {self.best_energy:0.6f}\n')
        print(f'  final energy: {self.best_energy}\n')

        print('+-------------------------- END ---------------------------+')

    # linear multiplicative cooling
    def cooling_linear_m(self, step):
        return self.t_max /  (1 + self.alpha * step)

    # linear additive cooling
    def cooling_linear_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step)/self.step_max)

    # quadratic multiplicative cooling
    def cooling_quadratic_m(self, step):
        return self.t_min / (1 + self.alpha * step**2)

    # quadratic additive cooling
    def cooling_quadratic_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step)/self.step_max)**2

    # exponential multiplicative cooling
    def cooling_exponential_m(self, step):
        return self.t_max * self.alpha**step

    # logarithmical multiplicative cooling
    def cooling_logarithmic_m(self, step):
        return self.t_max / (self.alpha * log(step + 1))


    def safe_exp(self, x):
        try: return exp(x)
        except: return 0


#--- END ----------------------------------------------------------------------+





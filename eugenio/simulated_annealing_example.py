# taken from https://github.com/nathanrooy/simulated-annealing


from simulated_annealing import sa
from landscapes.single_objective import sphere
from landscapes.single_objective import tsp


#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
# Continous Optimization
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

x0 = [1, 2, 3]
opt1 = sa.minimize(sphere, x0, opt_mode='continuous', step_max=1000, t_max=1, t_min=0)

opt1.results()



#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
# Combinatorial Optimization
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------



#For combinatorial problems such as the traveling salesman problem, usage is just as easy. 
# First, let's define a method for calculating the distance between our points. In this case,
# Euclidean distance is used, but it can be anything...

def calc_euclidean(p1, p2):    
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

#Next, let's prepare some points. In the intrest of simplicity, we'll 
#just generate 6 points on the perimiter of the unit circle.

from math import cos
from math import sin
from math import pi

n_pts = 6
d_theta = (2 * pi) / n_pts
theta = [d_theta * i for i in range(0, n_pts)]
x0 = [(cos(r), sin(r)) for r in theta]


# Now, prepeare the cost function.

cost_func = tsp(calc_euclidean, close_loop=True).dist

# Now let's optimize this while remembering to shuffle the points prior to running.

from random import shuffle
shuffle(x0)
opt = sa.minimize(cost_func, x0, opt_mode='combinatorial', step_max=1000, t_max=1, t_min=0)

opt.results()
opt.best_state


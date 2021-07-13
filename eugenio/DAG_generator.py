"""
DAG generation

"""

"""
https://ipython.org/ipython-doc/stable/parallel/dag_dependencies.html

"""
import networkx as nx
import random as ran

#-------------------
def random_dag(nodes, edges):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = ran.randint(0,nodes-1)
        b=a
        while b==a:
            b = ran.randint(0,nodes-1)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a,b)


    
    
    dag_dict = {}
    
    for node in G:
        dag_dict[node] = [n for n in G.neighbors(node)]
    
    dag=dag_dict
    
    return G,dag

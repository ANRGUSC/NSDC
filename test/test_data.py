from bnet import TaskGraph, Network
from heft.heft import schedule_dag
import networkx as nx 
import pandas as pd 

def schedule(dag: nx.DiGraph, comp: pd.DataFrame, comm: pd.DataFrame) -> float:
    network_relabel = {node: i for i, node in enumerate(comm.columns)}
    task_relabel = {task: i for i, task in enumerate(comp.index.values)}
    nx.set_edge_attributes(dag, nx.get_edge_attributes(dag, "data"), name="weight")

    proc_sched, task_sched, dict_sched = schedule_dag(
        nx.relabel_nodes(dag, task_relabel),
        communication_matrix=comm.rename(
            index=network_relabel, 
            columns=network_relabel
        ).values,
        computation_matrix=comp.rename(
            index=task_relabel, 
            columns=network_relabel
        ).values
    )

    return max([event.end for _, event in task_sched.items()])


def main():
    tg = TaskGraph()
    tg.add_task("load", lambda x: x * 0.5)
    tg.add_task("process1", lambda x: x * 0.7)
    tg.add_task("process2", lambda x: x * 0.2)
    tg.add_task("agg", lambda x: x * 0.9)
    tg.add_dependency("load", "process2", 1)
    tg.add_dependency("load", "process1", 5)
    tg.add_dependency("process1", "agg", 6)
    tg.add_dependency("process2", "agg", 7)

    network = Network()
    network.add_node("node1", 5)
    network.add_node("node2", 5)
    network.add_node("node3", 5)
    network.add_edge("node1", "node2", 1)
    network.add_edge("node2", "node3", 2)



    res = schedule(
        tg.to_networkx(),
        tg.computation_matrix(network),
        network.communication_matrix()
    )

    print(res)


if __name__ == "__main__":
    main()
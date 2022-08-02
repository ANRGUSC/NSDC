import pathlib
from typing import Callable, Dict, Hashable, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from nsdc.generators.generator import TaskGraphGenerator
from nsdc.network.network import Network
from nsdc.network.simple_network import SimpleNetwork
from nsdc.task_graph.simple_task_graph import SimpleTaskGraph
from nsdc.task_graph.task_graph import TaskGraph
from wfcommons.common.task import Task
from wfcommons.common.workflow import Workflow
from wfcommons.wfchef.recipes import MontageRecipe

from heft_synthesis import draw_schedule, heft, schedule_to_dataframe

thisdir = pathlib.Path(__file__).parent.resolve()

class WorkflowTaskGraph(TaskGraph):
    def __init__(self, workflow: Workflow) -> None:
        super().__init__()
        self._graph = nx.DiGraph()

        data: Dict[str, Dict[str, Union[str, Set[str]]]] = {}
        for node in workflow.nodes:
            task: Task = workflow.nodes[node]['task']
            self.add_task(task.name, task.runtime)
            for file in task.files:
                data.setdefault(file.name, {'source': None, 'destinations': set()})
                if file.link == 'output':
                    if 'source' in data[file.name]:
                        raise ValueError(f'File {file.name} has multiple sources')
                    data[file.name]['source'] = task.name
                else:
                    data[file.name]['destinations'].add(task.name)
     
        for src, dst in workflow.edges:
            src_task: Task = workflow.nodes[src]['task']
            dst_task: Task = workflow.nodes[dst]['task']

            dep_size: float = 0.0
            for file in src_task.files:
                if file.link == 'output' and dst_task.name in data[file.name]['destinations']:
                    dep_size += file.size

            self.add_dependency(src_task.name, dst_task.name, dep_size)

    def add_task(self, name: str, cost: Callable[[float], float]) -> None:
        self._graph.add_node(name, cost=cost)

    def remove_task(self, name: str) -> None:
        self._graph.remove_node(name)
    
    def add_dependency(self, src: str, dst: str, data: float) -> None:
        self._graph.add_edge(src, dst, data=data)

    def remove_dependency(self, src: str, dst: str) -> None:
        self._graph.remove_edge(src, dst)

    def computation_matrix(self, 
                           network: Network, 
                           can_execute: Callable[[Hashable, Hashable], bool] = lambda *_: True) -> pd.DataFrame:
        network_graph = network.to_networkx()
        rows = []
        for task in self._graph.nodes:
            task_cost: float = self._graph.nodes[task]["cost"]
            cells = []
            for node in network_graph.nodes:
                node_speed: float = network_graph.nodes[node]["speed"]
                cannot_execute = node_speed == 0 or not can_execute(task, node)
                if cannot_execute:
                    cells.append(np.inf)
                else:
                    cells.append(task_cost / node_speed)
            rows.append(cells)

        return pd.DataFrame(
            rows,
            columns=list(network_graph.nodes),
            index=list(self._graph.nodes)
        )

    def data_matrix(self) -> pd.DataFrame:
        rows = []
        for task in self._graph.nodes:
            cells = []
            for dep in self._graph.nodes:
                if self._graph.has_edge(task, dep):
                    cells.append(self._graph.edges[task, dep]['data'])
                else:
                    cells.append(0.0)
            rows.append(cells)

        return pd.DataFrame(
            rows,
            columns=list(self._graph.nodes),
            index=list(self._graph.nodes)
        )

    def to_networkx(self) -> nx.DiGraph:
        return self._graph

class WorkflowGenerator(TaskGraphGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate(self) -> SimpleTaskGraph:
        task_graph = SimpleTaskGraph(
            task_cost={
                SimpleTaskGraph.Cost.LOW: 1,
                SimpleTaskGraph.Cost.HIGH: 2,
            },
            data_cost={
                SimpleTaskGraph.Cost.LOW: 1,
                SimpleTaskGraph.Cost.HIGH: 2,
            }
        )
        
        recipe = MontageRecipe(num_tasks=100)
        workflow = recipe.build_workflow()


        return workflow


def get_network() -> SimpleNetwork:
    ## MOTHER NETWORK ##
    # Construct Network
    mother_network = SimpleNetwork(
        node_speed={
            # Node speed, a node with speed s computes a task with size t
            # in n/s seconds
            SimpleNetwork.Speed.LOW: 1/10,
            SimpleNetwork.Speed.HIGH: 2/10,
        },
        sat_speed={
            # Satellite speed, an edge with speed (transfer rate) s transmits
            # data of size d between two nodes in s/d seconds
            SimpleNetwork.Speed.LOW: 1/5,
            SimpleNetwork.Speed.HIGH: 2/5,
        },
        radio_speed={
            SimpleNetwork.Speed.LOW: 1/5,
            SimpleNetwork.Speed.HIGH: 2/5,
        },
        gray_speed={
            SimpleNetwork.Speed.LOW: 4/5,
            SimpleNetwork.Speed.HIGH: 8/5,
        },

        # Deploy Cost by resource
        node_cost={
            SimpleNetwork.Speed.LOW: 1,
            SimpleNetwork.Speed.HIGH: 2,
        },
        sat_cost={
            SimpleNetwork.Speed.LOW: 5,
            SimpleNetwork.Speed.HIGH: 10,
        },
        radio_cost={
            SimpleNetwork.Speed.LOW: 5,
            SimpleNetwork.Speed.HIGH: 10,
        },
        gray_cost={
            SimpleNetwork.Speed.LOW: 1,
            SimpleNetwork.Speed.HIGH: 2,
        }
    )
    # Add 5 nodes with specified compute speeds and positions
    mother_network.add_node(0, SimpleNetwork.Speed.LOW, pos=(0, 0.25))
    mother_network.add_node(1, SimpleNetwork.Speed.LOW, pos=(0.35, 0.3))
    mother_network.add_node(2, SimpleNetwork.Speed.LOW, pos=(0.4, 0.15))
    mother_network.add_node(3, SimpleNetwork.Speed.HIGH, pos=(0.8, 0.3))
    mother_network.add_node(4, SimpleNetwork.Speed.HIGH, pos=(0.6, 0.6))

    # Add gray network edges
    mother_network.add_gray_edge(2, 3, SimpleNetwork.Speed.HIGH)
    mother_network.add_gray_edge(3, 4, SimpleNetwork.Speed.HIGH)
    mother_network.add_gray_edge(2, 4, SimpleNetwork.Speed.LOW)

    # Add radio network edges
    mother_network.add_radio_edge(1, 0, SimpleNetwork.Speed.LOW)
    mother_network.add_radio_edge(2, 1, SimpleNetwork.Speed.HIGH)
    mother_network.add_radio_edge(2, 0, SimpleNetwork.Speed.LOW)
    mother_network.add_radio_edge(2, 3, SimpleNetwork.Speed.LOW)

    # Add satellite edges
    mother_network.add_satellite_edge(1, SimpleNetwork.Speed.LOW)
    mother_network.add_satellite_edge(4, SimpleNetwork.Speed.LOW)
    # mother_network.add_satellite_edge(2, SimpleNetwork.Speed.HIGH)

    return mother_network


def main():
    workflow = MontageRecipe(num_tasks=100).build_workflow()
    task_graph = WorkflowTaskGraph(workflow)
    network = get_network()

    task_order = list(nx.topological_sort(task_graph.to_networkx()))
    node_order = network.nodes

    comp = task_graph.computation_matrix(network).loc[task_order,node_order]
    data = task_graph.data_matrix().loc[task_order,task_order]
    comm = network.communication_matrix().loc[node_order,node_order]

    def compare(selected_nodes: Set[int],
                node: Tuple[int, float], 
                other: Tuple[int, float]) -> Union[float, int]:
        node_idx, node_eft = node
        other_idx, other_eft = other
        eft_diff = node_eft - other_eft

        selected_names = [node_order[idx] for idx in selected_nodes]
        node_cost = 0.0
        for n in [node_order[node_idx], *selected_names]:
            if n == "__satellite__":
                continue
            node_cost += network.node_cost[network._graph.nodes[n]["speed"]]

        other_cost = 0.0
        for n in [node_order[other_idx], *selected_names]:
            if n == "__satellite__":
                continue
            other_cost += network.node_cost[network._graph.nodes[n]["speed"]]

        cost_diff = node_cost - other_cost
        # print(eft_diff + cost_diff)
        return eft_diff + cost_diff

    schedule = heft(data.values, comp.values.T, comm.values, compare=compare)
    df = schedule_to_dataframe(schedule).sort_values(["Start"])
    print(df.to_string(index=False))
    fig = draw_schedule(schedule)
    fig.write_image(thisdir.joinpath("schedule.png"))

    # dataset = [
    #     get_workflow() for i in range(50)
    # ]

    # network = get_network()

    # _generator = ExampleGenerator([5, 3, 1])
    # task_graph_generator = TaskGraphSetGenerator(
    #     task_graphs=[
    #         _generator.generate()
    #         for _ in range(NUM_TASK_GRAPHS)
    #     ]
    # )
    # def cost_func(network: SimpleNetwork) -> Result:
    #     task_graphs = []
    #     makespans = []
    #     for _ in range(NUM_TASK_GRAPHS):
    #         task_graphs.append(task_graph_generator.generate())
    #         try:
    #             makespan = scheduler.schedule(
    #                 network=network,
    #                 task_graph=task_graphs[-1]
    #             )
    #         except AssertionError:
    #             continue
    #         makespans.append(makespan)

    #     makespan = np.inf if not makespans else sum(makespans) / len(makespans)
    #     deploy_cost = network.cost()
    #     risk = network.risk()
    #     return Result(
    #         network=network,
    #         cost=(
    #             COEF_DEPLOY_COST * deploy_cost + 
    #             COEF_RISK * risk +
    #             COEF_MAKESPAN * makespan
    #         ),
    #         metadata={
    #             "task_graphs": task_graphs,
    #             "makespans": makespans,
    #             "deploy_cost": deploy_cost,
    #             "risk": risk,
    #         }
    #     )

    # SimulatedAnnealingOptimizer(
    #     start_network=network.random_subnetwork(),
    #     cost_func=
    # )


if __name__  == '__main__':
    main()

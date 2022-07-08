import pathlib
from datetime import datetime, timedelta
from functools import partial
from random import randint
from typing import (Callable, Dict, Hashable, Iterable, List, Set, Tuple,
                    TypeVar, Union)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

thisdir = pathlib.Path(__file__).resolve().parent

ScheduleType = Dict[Hashable, List[Tuple[Hashable, float, float]]]

Comparable = TypeVar('Comparable')
Comparator = Callable[[Comparable, Comparable], Union[float, int]]
def min_compare(values: Iterable[Comparable], 
                compare: Comparator) -> Comparable:
    """Finds the minimum element in linear time with a custom compare function

    Args:
        values: values to get minimum value of
        compare: compare function to compare two elements of values. This \
            function should take two elements and return 0 if they are equal \
            a value less than 0 if the first argument is less than the second \
            argument, and a value greater than 0 otherwise.
    
    Returns:
        Comparable: minimum element in values with respect to compare function
    """
    vals = values
    while len(vals) > 1:
        cmp_val = vals[randint(0, len(vals)-1)]
        vals = [val for val in vals if compare(val, cmp_val) < 0]
        if not vals:
            return cmp_val
    return vals[0]

def argmin_compare(values: Iterable[Comparable], compare: Comparator) -> int:
    """Finds the index of the minimum element in linear time with a custom \
        compare function
    
    Args:
        values: values to get minimum value of
        compare: compare function to compare two elements of values. This \
            function should take two elements and return 0 if they are equal \
            a value less than 0 if the first argument is less than the second \
            argument, and a value greater than 0 otherwise.
    
    Returns:
        int: index of minimum element in values with respect to compare function
    """
    minval = min_compare(values, compare)
    for val in values:
        if compare(val, minval) == 0:
            return 0
    raise Exception(f"Invalid minval={minval} for values={values}")

def default_compare(selected_nodes: Set[int],
                    node: Tuple[int, float], 
                    other: Tuple[int, float]) -> Union[float, int]:
    """Default comparison function for HEFT

    Compares the earliest finish times (EFT) of nodes
    
    Args:
        selected_nodes: Nodes which have already been scheduled tasks to run
        node: tuple of (ID, EFT) for the first node
        other: tuple of (ID, EFT) for the second node
    
    Returns:
        Union[float, int]: 0 if nodes are equal a value less than 0 if the \
            first node is less than the second argument, and a value greater \
            than 0 otherwise.
    """
    _, node_eft = node
    _, other_eft = other
    return node_eft - other_eft

NodeComparator = Callable[
    [Set[int], Tuple[int, float], Tuple[int, float]], 
    Union[float, int]
]

def heft(data: np.ndarray, 
         comp: np.ndarray, 
         comm: np.ndarray,
         compare: NodeComparator = default_compare) -> ScheduleType:
    """HEFT algorithm with custom node comparison

    Finds a schedule for a network with n nodes and a task graph with m tasks.

    Args:
        data: (m x m) matrix of intertask communication data. data[i][j] \
            represents the amount of data which is output from task i and \
            input task j
        comp: (n x m) matrix of computation cost of each task on each node. \
            comp[i][j] is the time it takes to compute task j on node i.
        comm: (n x n) matrix of communication costs for every pair of nodes. \
            It would take comm[i][j] seconds per unit of data to transfer data \
            between nodes i and j.
        compare: comparison function to decide which node to schedule a task \
            on. The function should take three arguments: 1) a set of nodes \
            which currently have at least one task scheduled on them 2) tuple \
            of (ID, EFT) for the first node 3) tuple of (ID, task estimated \
            finish time (EFT)) for the second node. The function should return \
            0 if the nodes are equal, a value less than 0 if the first node is \
            less than the second argument, and a value greater than 0 otherwise.
    """
    num_nodes, num_tasks = comp.shape
    assert(comm.shape == (num_nodes, num_nodes))
    assert(data.shape == (num_tasks, num_tasks))

    w = np.mean(comp, axis=0)
    c = data / np.mean(comm)

    rank = np.zeros(num_tasks)
    for task in reversed(range(num_tasks)):
        idx = np.nonzero(data[task])[0]        
        rank[task] = w[task] + (0 if len(idx) == 0 else np.max(c[task][idx] + rank[idx]))

    finished = np.zeros(num_tasks, dtype=np.float) + np.inf
    schedule = np.zeros(num_tasks, dtype=int) - 1

    schedules = {p: [(-1, 0, 0), (-1, np.inf, np.inf)] for p in range(num_nodes)}
    for task in np.argsort(rank)[::-1]: # O(n)
        deps_idx = np.nonzero(data[:,task])[0]
        insert = {}
        for proc in range(num_nodes): # O(m)
            start = 0 if len(deps_idx) <= 0 else np.max(finished[deps_idx] + data[:,task][deps_idx] * comm[proc,schedule[deps_idx]])
            for i in range(len(schedules[proc]) - 1): # O(m)
                gap_open, gap_close = schedules[proc][i][2], schedules[proc][i+1][1]
                if start > gap_close:
                    continue
                actual_start = max(start, gap_open)
                if gap_close - actual_start >= comp[proc][task]:
                    insert[proc] = (i+1, actual_start, actual_start + comp[proc][task])
                    break
        
        vals = [(proc, eft) for proc, (_, _, eft) in insert.items()]
        if len(vals) == 0:
            print(schedules, flush=True)
            print(finished, flush=True)
            print(schedule, flush=True)
            raise ValueError(f"No valid schedule for task {task}")
        sched_proc, _  = min_compare( # O(m)
            vals,
            compare=partial(compare, set(schedule))
        )
        schedules[sched_proc].insert(insert[sched_proc][0], (task, *insert[sched_proc][1:])) # O(m)
        finished[task] = insert[sched_proc][2]
        schedule[task] = sched_proc

    return {
        proc: tasks[1:-1]
        for proc, tasks in schedules.items()
    }

def schedule_to_dataframe(schedule: ScheduleType) -> pd.DataFrame:
    """Converts a HEFT schedule to a pndas dataframe
    Args:
        schedule: schedule to convert

    Returns:
        pd.DataFrame: schedule as a dataframe
    """
    return pd.DataFrame(
        [
            [proc, task_id, start, end]
            for proc, tasks in schedule.items()
            for task_id, start, end in tasks
        ],
        columns=["Processor", "Task", "Start", "Finish"]
    )

def draw_schedule(schedule: ScheduleType) -> go.Figure:
    """Draws a schedule as a Gantt chart

    Args:
        schedule: schedule to draw
    
    Returns:
        go.Figure: plotly Figure object of schedule Gantt chart
    """
    df = schedule_to_dataframe(schedule)

    df["Processor"] = df["Processor"].astype(str)
    df["Task"] = df["Task"].astype(str)
    df["Start Date"] = df["Start"].apply(lambda x: datetime.now() + timedelta(seconds=x))
    df["Finish Date"] = df["Finish"].apply(lambda x: datetime.now() + timedelta(seconds=x))
    
    fig = px.timeline(
        df,
        x_start="Start Date",
        x_end="Finish Date",
        y="Processor",
        color="Task",
        hover_data={
            "Task": True, 
            "Start": True, 
            "Finish": True,
            "Start Date": False,
            "Finish Date": False,
            "Processor": False
        }
    )
    fig.update_layout({"xaxis_tickformat": '%X.%LS'})
    return fig

def main():
    data = np.array([
    # DST 0   1      2      3     4
        [0.0, 10.0, 10.0, 0.0,  0.0],  # SRC 0
        [0.0, 0.0,   5.0, 5.0,  0.0],  # SRC 1
        [0.0, 0.0,   0.0, 0.0,  1.0],  # SRC 2
        [0.0, 0.0,   0.0, 0.0,  3.2],  # SRC 3
        [0.0, 0.0,   0.0, 0.0,  0.0],  # SRC 4
    ])

    comp = np.array([
    # TASK 0  1    2    3    4                
        [10.0, 20.0, 10.0, 13.0, 50.0], # NODE 1
        [10.2, 10.0, 10.3, 12.3, 10.3], # NODE 2
        [10.0, 20.0, 20.0, 11.2, 10.3], # NODE 3
    ]) 

    comm = np.array([
        [0.0, 1.4, 2.0],
        [1.4, 0.0, 4.0],
        [2.0, 4.0, 0.0],
    ])

    costs = {
        0: 10,
        1: 100,
        2: 500
    }

    def compare(selected_nodes: Set[int],
                node: Tuple[int, float], 
                other: Tuple[int, float]) -> float:
        node_id, node_eft = node
        other_id, other_eft = other

        if node_id not in selected_nodes:
            node_eft += costs[node_id]
        if other_id not in selected_nodes:
            other_eft += costs[other_id]
        
        return default_compare(
            selected_nodes, 
            (node_id, node_eft), 
            (other_id, other_eft)
        )

    schedule = heft(data, comp, comm, compare=compare)
    df = schedule_to_dataframe(schedule).sort_values(["Start"])
    print(df.to_string(index=False))
    fig = draw_schedule(schedule)
    fig.write_image(thisdir.joinpath("schedule.png"))

if __name__ == "__main__":
    main()

import base64
from functools import lru_cache

import numpy as np
from typing import List, Tuple
import dash_bootstrap_components as dbc
from dash import Input, Output, Dash, dcc, html, no_update
import plotly.express as px

from bnet.network.simple_network import SimpleNetwork
from bnet.optimizers.optimizer import Result

import pathlib 
import dill as pickle 
import io 
import matplotlib.pyplot as plt 
import pandas as pd 

homedir = pathlib.Path.home()
thisdir = pathlib.Path(__file__).resolve().parent
pickle_path = homedir.joinpath(".bnet")

mother_network: SimpleNetwork = pickle.loads(
    pickle_path.joinpath("mother.pickle").read_bytes()
)

image_style = {
    "width": "100%",
    "object-fit": "cover",
    # "height": "300px",
}

def load() -> List[Result]:
    return pickle.loads(pickle_path.joinpath("result.pickle").read_bytes())

@lru_cache(maxsize=None)
def load_metrics(_rand: int = -1) -> Tuple[List[Result], pd.DataFrame]:
    results = load()
    metrics = pd.DataFrame.from_records([
        {
            "cost": result.cost, 
            "makespan": np.mean(result.metadata["makespans"]),
            "deploy_cost": result.metadata["deploy_cost"],
            "risk": result.metadata["risk"],
            "seq": result.metadata["seq"]
        } 
        for result in results
    ])
    return results, metrics

def fig_to_base64(fig: plt.Figure) -> str:
    bio = io.BytesIO()
    plt.savefig(bio, format="png")
    bio.seek(0)
    return f"data:image/png;base64,{base64.b64encode(bio.read()).decode()}"

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

def layout():
    return dbc.Container(
        [
            # dcc.Interval(id="interval", interval=2000),
            # dbc.Row(
            #     [ 
            #         dbc.Col(id="network", xs=4),
            #         dbc.Col(id="task_graph", xs=4),
            #         dbc.Col(id="best_network", xs=4),
            #     ]
            # ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("update", id="update-button", block=True),
                        className="mt-4 mb-4"
                    )
                ]
            ),
            dbc.Row(
                [ 
                    dbc.Col(dcc.Graph(id="plot")),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(id="chosen_network", className="img-responsive", style=image_style),
                    dbc.Col(id="chosen_task_graph", className="img-responsive", style=image_style),
                ]
            )
        ],
        fluid=True
    )

app.layout = layout

@app.callback(
    # Output("network", "children"),
    # Output("task_graph", "children"),
    # Output("best_network", "children"),
    Output("plot", "figure"),
    Input("update-button", "n_clicks")
)
def network_callback(n_clicks: int) -> List:
    # results = load() 
    # if len(results) < 0 or n_intervals is None:
    #     result = Result()
    # else:
    #     result = results[min(n_intervals, len(results) - 1)]

    # network, task_graph, best_network, plot_fig = None, None, None, no_update

    # # if result.last_network:
    # #     fig, _ = mother_network.draw(result.last_network.edges)
    # #     network = html.Img(src=fig_to_base64(fig))
    # #     plt.close(fig)

    # # if result.last_task_graph:
    # #     task_graph: SimpleTaskGraph = result.last_task_graph
    # #     fig, _ = task_graph.draw()
    # #     task_graph = html.Img(src=fig_to_base64(fig))
    # #     plt.close(fig)

    # # if result.best_network:
    # #     fig, _ = mother_network.draw(result.best_network.edges)
    # #     best_network = html.Img(src=fig_to_base64(fig))
    # #     plt.close(fig)

    _, metrics = load_metrics(n_clicks)
    metrics = metrics.drop(columns=["cost"])
    # metrics = metrics.groupby(["deploy_cost", "risk"]).mean().reset_index()
    plot_fig = px.scatter(
        metrics, 
        x="deploy_cost", 
        y="makespan", 
        color="risk",
        template="simple_white",
        color_continuous_scale="RdYlBu_r"
    )
    plot_fig.update_traces(marker=dict(size=12))

    # return network, task_graph, best_network, plot_fig
    return plot_fig

@app.callback(
    Output("chosen_network", "children"),
    # Output("chosen_task_graph", "children"),
    Input("plot", "clickData")
)
def click_callback(click_data):
    if not click_data:
        return no_update
    point = click_data["points"][0]

    results, metrics = load_metrics()

    idx = metrics[
        (metrics["deploy_cost"] == point["x"]) & 
        (metrics["makespan"] == point["y"]) & 
        (metrics["risk"] == point["marker.color"])
    ].index[0]
    
    result = results[idx]
    if result.network:
        fig, _ = mother_network.draw(result.network.edges)
        network = html.Img(src=fig_to_base64(fig))
        plt.close(fig)
    
    # if result.last_task_graph:
    #     fig, _ = result.last_task_graph.draw()
    #     task_graph = html.Img(src=fig_to_base64(fig))
    #     plt.close(fig)

    return network #, task_graph


if __name__ == "__main__":
    app.run_server(debug=True)

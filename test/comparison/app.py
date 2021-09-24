import base64

import numpy as np
from typing import List, Tuple
import dash_bootstrap_components as dbc
from dash import Input, Output, Dash, dcc, html, no_update
import plotly.express as px

from bnet.network.simple_network import SimpleNetwork

import pathlib 
import dill as pickle 
import io 
import matplotlib.pyplot as plt 
import pandas as pd 

thisdir = pathlib.Path(__file__).resolve().parent

mother_network: SimpleNetwork = pickle.loads(
    thisdir.joinpath("mother.pickle").read_bytes()
)

image_style = {
    "width": "100%",
    "object-fit": "cover",
    # "height": "300px",
}

def load_metrics() -> pd.DataFrame:
    bf_results = pickle.loads(thisdir.joinpath("bf_result.pickle").read_bytes())
    sa_results = pickle.loads(thisdir.joinpath("sa_result.pickle").read_bytes())
    rows = []
    for optimizer, results in {"brute_force": bf_results, "simulated_annealing": sa_results}.items():
        for result in results:
            rows.append([
                result.cost, 
                np.mean(result.metadata["makespans"]),
                result.metadata["deploy_cost"],
                result.metadata["risk"],
                result.metadata["seq"],
                optimizer
            ])
    return pd.DataFrame(rows, columns=["cost", "makespan", "deploy_cost", "risk", "seq", "optimizer"])

def fig_to_base64(fig: plt.Figure) -> str:
    bio = io.BytesIO()
    plt.savefig(bio, format="png")
    bio.seek(0)
    return f"data:image/png;base64,{base64.b64encode(bio.read()).decode()}"

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

def layout():
    return dbc.Container(
        [
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
    Output("plot", "figure"),
    Input("update-button", "n_clicks")
)
def network_callback(n_clicks: int) -> List:
    metrics = load_metrics()
    metrics = metrics.sort_values("seq")

    print(metrics)

    bf_metrics = metrics.loc[metrics["optimizer"] == "brute_force"]
    bf_metrics["cost"] = bf_metrics["cost"].cummin()
    
    sa_metrics = metrics.loc[metrics["optimizer"] == "simulated_annealing"]
    sa_metrics["cost"] = sa_metrics["cost"].cummin()

    metrics = pd.concat([bf_metrics, sa_metrics])
    plot_fig = px.line(
        metrics, 
        x="seq", 
        y="cost", 
        color="optimizer",
        template="simple_white",
        title="Best Cost Over Time",
        labels={
            "cost": "Best Cost",
            "seq": "Iteration",
            "optimizer": "Optimizer"
        }
    )
    plot_fig.update_traces(marker=dict(size=12))
    return plot_fig


if __name__ == "__main__":
    app.run_server(debug=True)

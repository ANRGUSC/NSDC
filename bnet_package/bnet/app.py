import json
from typing import Dict, List, Optional, Tuple, Union
import dash
from dash.dependencies import Input, State, Output
import dash_cytoscape as cyto
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from uuid import uuid4

import numpy
from bnet.network import MixedNetwork, SimpleNetwork

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX]
)

graph = cyto.Cytoscape(
    id="cyto-graph",
    layout={'name': 'preset'},
    style={'width': '100%', 'height': '800px'},
    elements=[],
    stylesheet=[
        # Group selectors
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)'
            }
        },

        # Class selectors
        {
            'selector': '.satellite',
            'style': {
                'shape': 'triangle',
                'line-color': 'green',
            }
        },
        {
            'selector': '.radio',
            'style': {
                'line-color': 'blue',
                'line-style': 'dotted'
            }
        },
        {
            'selector': '.gray',
            'style': {
                'line-color': 'gray',
            }
        }
    ]
)

simple_graph = cyto.Cytoscape(
    id="cyto-simple-graph",
    layout={'name': 'preset'},
    style={'width': '100%', 'height': '800px'},
    elements=[],
    stylesheet=[
        # Group selectors
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)'
            }
        },

        # Class selectors
        {
            'selector': '.satellite',
            'style': {
                'shape': 'triangle',
                'line-color': 'green',
            }
        },
        {
            'selector': '.gray',
            'style': {
                'line-color': 'gray',
            }
        },
        {
            'selector': '.radio',
            'style': {
                'line-color': 'blue',
                'line-style': 'dotted'
            }
        }
    ]
)



# def get_form(network: MixedNetwork, data: List[Dict]) -> Tuple[List, Dict]:
#     router_bandwidth = None 
#     node_speed = None
#     node_satellite_bandwidth = None
#     internet_edge_bandwidth = None
#     radio_edge_bandwidth = None
#     if len(data) == 1:
#         node = data[0]["id"]
#         if node in network._routers: # ROUTER
#             title = "Router Options"
#             data = {"mode": "router", "node": node}
#             router_bandwidth = network.get_router_bandwidth(node)
#         else: # NODE
#             title = "Node Options"
#             data = {"mode": "node", "node": node}
#             node_speed = network.get_node_value(node)
#             node_satellite_bandwidth = network.get_satellite_connection(node)
#     elif len(data) == 2:
#         node = data[0]["id"]
#         other = data[1]["id"]
#         if node in network._routers or other in network._routers: # INTERNET EDGE
#             title = "Internet Edge Options"
#             data = {"mode": "internet_edge", "node": node, "other": other}
#             internet_edge_bandwidth = network.get_internet_edge_bandwidth(node, other)
#         else:
#             title = "Radio Edge Options"
#             data = {"mode": "radio_edge", "node": node, "other": other}
#             radio_edge_bandwidth = network.get_radio_edge_bandwidth(node, other)
#     else:
#         title = "Select a node, pair of nodes, or an edge"
#         data = {"mode": "no_selection"}

#     options = [
#         dbc.FormGroup(
#             [
#                 dbc.Label("Bandwidth"),
#                 dbc.Input(id='in-router-bandwidth', value=router_bandwidth, type="numeric")
#             ],
#             style={"display": "none"} if router_bandwidth is None else {}
#         ),
#         dbc.FormGroup(
#             [
#                 dbc.Label("Speed"),
#                 dbc.Input(id='in-node-speed', value=node_speed, type="numeric")
#             ],
#             style={"display": "none"} if node_speed is None else {}
#         ),
#         dbc.FormGroup(
#             [
#                 dbc.Label("Satellite Bandwidth"),
#                 dbc.Input(id='in-node-satellite-bandwidth', value=node_satellite_bandwidth, type="numeric")
#             ],
#             style={"display": "none"} if node_satellite_bandwidth is None else {}
#         ),
#         dbc.FormGroup(
#             [
#                 dbc.Label("Bandwidth"),
#                 dbc.Input(id='in-internet-edge-bandwidth', value=internet_edge_bandwidth, type="numeric")
#             ],
#             style={"display": "none"} if internet_edge_bandwidth is None else {}
#         ),
#         dbc.FormGroup(
#             [
#                 dbc.Label("Bandwidth"),
#                 dbc.Input(id='in-radio-edge-bandwidth', value=radio_edge_bandwidth, type="numeric")
#             ],
#             style={"display": "none"} if radio_edge_bandwidth is None else {}
#         ),
#         dbc.FormGroup(
#             dbc.Button("Submit", id="btn-submit", color="primary"),
#             style={"display": "none"} if data["mode"] == "no_selection" is None else {}
#         )
#     ]

#     form_groups = [
#         dbc.Row(dbc.Col(html.H5(title, id="form-title"))),
#         dbc.Row(dbc.Col(dbc.Form(options)))
#     ]
#     return form_groups, data

# default_form, _ = get_form(MixedNetwork(), {})
app.layout = dbc.Container(
    [
        dcc.Store(id="network-json"),
        dcc.Store(id="form-mode"),
        dcc.Store(id="last-click"),
        
        dbc.Row(
            [
                dbc.Col(graph),
                dbc.Col(simple_graph),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button("Generate", id="btn-generate", color="primary", block=True)
                )
            ]
        )
    ],
    fluid=True
)
import networkx as nx 

@app.callback(
    Output("cyto-graph", "elements"), 
    Output("cyto-simple-graph", "elements"), 
    Input("btn-generate", "n_clicks")
)
def update_graph_callback(n_clicks: str) -> List:
    if n_clicks is None:
        return dash.no_update, dash.no_update
    network = SimpleNetwork.random(zipf_constant=1.9)

    elements = [
        {
            "data": {"id": str(node), "type": "node"},
            "position": {
                "x": network._graph.nodes[node]["pos"][0]*300, 
                "y": network._graph.nodes[node]["pos"][1]*300
            },
            "classes": "satellite" if node == "__satellite__" else ""
        }
        for node in network.nodes
    ]
    elements += [
        {
            "data": {"source": str(src), "target": str(dst)},
            "classes": key
        }
        for src, dst, key in network.edges
    ]
    
    simple_graph = network.to_simple_networkx()
    simple_elements = [
        {
            "data": {"id": str(node), "type": "node"},
            "position": {
                "x": simple_graph.nodes[node]["pos"][0]*300, 
                "y": simple_graph.nodes[node]["pos"][1]*300
            },
            "classes": "satellite" if node == "__satellite__" else ""
        }
        for node in simple_graph.nodes
    ]
    simple_elements += [
        {
            "data": {"source": str(src), "target": str(dst)},
        }
        for src, dst in simple_graph.edges
    ]
    return elements, simple_elements















task_graph_generator.generate()






# @app.callback(
#     Output("network-json", "data"), 
#     Output("last-click", "data"),
#     Input("btn-add-router", "n_clicks_timestamp"),
#     Input("btn-add-node", "n_clicks_timestamp"),
#     Input("btn-submit", "n_clicks_timestamp"),
#     State("last-click", "data"),
#     State("network-json", "data"),       
#     State('form-mode', 'data'), 
#     State("in-router-bandwidth", "value"),
#     State("in-node-speed", "value"),
#     State("in-node-satellite-bandwidth", "value"),
#     State("in-internet-edge-bandwidth", "value"),
#     State("in-radio-edge-bandwidth", "value")
# )
# def add_node_callback(router_click: int, 
#                       node_click: int,  
#                       submit_click: int, 
#                       last_click: int,
#                       network_json: str,
#                       form_mode_data: str,
#                       router_bandwidth: float,
#                       node_speed: float,
#                       node_satellite_bandwidth: float,
#                       internet_edge_bandwidth: float,
#                       radio_edge_bandwidth: float) -> List:
#     if not router_click and not node_click and not submit_click:
#         return dash.no_update, dash.no_update
#     clicks = [router_click or 0, node_click or 0, submit_click or 0]
#     click = numpy.argmax(clicks)
#     if clicks[click] <= (last_click or 0):
#         return dash.no_update, dash.no_update
#     network = MixedNetwork() if not network_json else MixedNetwork.from_json(network_json)
#     if click == 0: # Add router
#         network.add_router(uuid4().hex, 0)
#     elif click == 1: # Add Node
#         network.add_node(uuid4().hex, 0)
#     elif click == 2: # Submit Update
#         if not form_mode_data:
#             return dash.no_update, clicks[click]
#         form_mode = json.loads(form_mode_data)
#         # if form_title
#         if form_mode["mode"] == "router":
#             router = form_mode["node"]
#             if not router in network._routers:
#                 raise ValueError(f"No router in network '{router}'")
#             network._routers[router] = router_bandwidth
#         elif form_mode["mode"] == "node":
#             node = form_mode["node"]
#             if not node in network._nodes:
#                 raise ValueError(f"No node in network '{node}'")
#             network._nodes[node] = node_speed
#             network.add_satellite_connection(node, node_satellite_bandwidth)
#         elif form_mode["mode"] == "internet_edge":
#             node, other = form_mode["node"], form_mode["other"]
#             try:
#                 network.remove_internet_edge(node, other)
#             except ValueError:
#                 pass 
#             network.add_internet_edge(node, other, internet_edge_bandwidth)
#         elif form_mode["mode"] == "radio_edge":
#             node, other = form_mode["node"], form_mode["other"]
#             try:
#                 network.remove_radio_edge(node, other)
#             except ValueError:
#                 pass 
#             network.add_radio_edge(node, other, radio_edge_bandwidth)
#         else:
#             print(f"Unsupported mode:", form_mode["mode"])
#             return dash.no_update, clicks[click]
        
#     return network.to_json(), clicks[click]

# @app.callback(
#     Output('selection-form', 'children'),   
#     Output('form-mode', 'data'), 
#     Input('cyto-graph', 'selectedNodeData'),
#     State("network-json", "data")
# )
# def select_node_callback(data, network_json: str):
#     if not network_json:
#         return dash.no_update, dash.no_update
#     network = MixedNetwork.from_json(network_json)
#     if data is None:
#         return dash.no_update, dash.no_update
#     form, form_mode = get_form(network, data)
#     # if len(data) == 1:
#     #     form, data = get_form(network, data[0]["id"])
#     # elif len(data) == 2:
#     #     form, data = get_form(network, data[0]["id"], data[1]["id"])
#     # else:
#     #     form, data = no_selection, None
    
#     return form, json.dumps(form_mode)


if __name__ == '__main__':
    app.run_server(debug=True)
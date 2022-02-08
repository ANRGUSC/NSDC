import plotly.express as px 
import pandas as pd 
import pathlib

thisdir = pathlib.Path(__file__).resolve().parent
savedir = thisdir.joinpath("outputs")

def main():
    results_path = savedir.joinpath("results.csv")
    data: pd.DataFrame = pd.read_csv(results_path, index_col=0)
    data = data.reset_index().rename(columns={data.index.name: "iteration"})

    # fig = px.scatter_3d(
    #     data,
    #     x="avg_makespan",
    #     y="deploy_cost",
    #     z="risk",
    #     color="cost",
    #     template="simple_white",
    #     color_continuous_scale="rdylbu_r",
    #     labels={
    #         "avg_makespan": "Average Makespan",
    #         "deploy_cost": "Deploy Cost",
    #         "risk": "Risk",
    #         "cost": "RDM Cost"
    #     }
    # )
    # fig.write_html(savedir.joinpath("all_costs_3d.html"))

    data["risk"] = data["risk"].astype(int).astype(str)
    color_scale = px.colors.diverging.RdYlBu
    num = len(data["risk"].unique())
    colors = [
        color_scale[i*len(color_scale)//num] for i in range(num)
    ]

    data["deploy_cost"] = data["deploy_cost"]**2.5

    fig = px.scatter(
        data,
        x="avg_makespan",
        y="cost",
        color="risk",
        size="deploy_cost",
        template="simple_white",
        color_discrete_sequence=colors,
        # title="Exhaustive Search",
        labels={
            "avg_makespan": "Average Makespan",
            "deploy_cost": "Deploy Cost",
            "risk": "Risk",
            "cost": "RDM Cost"
        },
        size_max=40
    ).update_traces(
        marker=dict(
            sizemode='area',
            line=dict(
                width=2,
                color="black"
            )
        )
    ).update_layout(
        legend={'traceorder':'reversed'}
    )
    fig.write_html(savedir.joinpath("all_costs.html"))
    fig.write_image(
        savedir.joinpath("all_costs.png")
    )







if __name__ == "__main__":
    main()
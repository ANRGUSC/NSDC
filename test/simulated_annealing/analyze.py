import pandas as pd 
import pathlib
import plotly.express as px

thisdir = pathlib.Path(__file__).resolve().parent
savedir = thisdir.joinpath("outputs")

def main():
    df = pd.read_csv(
        savedir.joinpath("results.csv"),
        index_col=None
    )
    
    print(df[["optimizer", "cost"]].groupby("optimizer").cummin())
    df["best_cost"] = df[["optimizer", "cost"]].groupby("optimizer").cummin()
    print(df)

    fig = px.line(
        df, 
        x="iteration", 
        y="best_cost", 
        color="optimizer",
        template="simple_white"
    )
    fig.write_image(savedir.joinpath("best_cost.png"))
    fig.write_html(savedir.joinpath("best_cost.html"))

    print(df)

if __name__ == "__main__":
    main()
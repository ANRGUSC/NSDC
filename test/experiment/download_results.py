import pandas as pd 
import wandb
from datetime import datetime 
import pathlib 

thisdir = pathlib.Path(__file__).resolve().parent

def main():
    api = wandb.Api()
    runs = api.runs(
        "anrg-iobt_ns/iobt_ns",
        filters={"createdAt": {"$gte": "2021-09-28T11:49:00"}}
    ) 

    IGNORE_KEYS = {"metadata", "mother_network", "network"}

    records = []
    for run in runs: 
        records.append({
            **{
                k: v for k, v in run.summary.items() 
                if not k.startswith("_") and k not in IGNORE_KEYS
            },
            **{
                k: v for k, v in run.config.items() 
                if not k.startswith("_") and k not in IGNORE_KEYS
            }
        })

    df = pd.DataFrame.from_records(records)
    print(df)

    df.to_csv(thisdir.joinpath("results.csv"))

if __name__ == "__main__":
    main()
import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import seaborn as sns
    import marimo as mo
    import pandas as pd
    import os
    import json
    return json, mo, np, os, pd, sns


@app.cell
def _():
    # lyrics_data |----- taylor_swift.json

    file_path = 'lyrics_data/taylor_swift.json'

    return (file_path,)


@app.cell
def _(file_path, pd):
    ts_df = pd.read_json(file_path, orient="index")
    return (ts_df,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

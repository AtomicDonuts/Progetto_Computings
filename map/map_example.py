import sys

from loguru import logger
import numpy as np
import plotly.express as px
import pandas as pd

sys.path.append("../imports/")
import custom_variables as custom_paths

def fig_generator(
        catalog_path=custom_paths.csv_path,
        input_dataframe=None,
        html_output=custom_paths.htmlmap_path
):
    '''
    '''
    if input_dataframe is None:
        logger.info(f"Loading from {catalog_path}")
        dataframe = pd.read_csv(catalog_path)
    else:
        if catalog_path != custom_paths.csv_path:
            logger.warning(
                "'input_dataframe' detected, 'catalog_path' will be ignored.")
        logger.info("Loading DataFrame from input")
        dataframe = input_dataframe

    fig = px.scatter_geo(
        dataframe,
        lat="LII",
        lon="BII",
        #color="type",
        hover_name="name",
        hover_data={
            "BII": False,
            "LII": False,
        },
        projection="mollweide",
        title="Sky Map",
        basemap_visible=False
    )

    fig.update_geos(showframe=True)

    fig.update_traces(
        marker=dict(
            size=5,
            opacity=0.8,
            line=dict(width=1)
        )
    )

    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        dragmode=False,
    )
    fig.write_html(html_output)
    return fig

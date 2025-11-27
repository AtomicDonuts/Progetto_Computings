'''
wip 
'''
import argparse
import sys

from loguru import logger
import plotly.express as px
import pandas as pd

sys.path.append("../imports/")
import custom_variables as custom_paths

def fig_generator(
        catalog_path=custom_paths.csv_path,
        input_dataframe=None,
        html_output=custom_paths.map_path
):
    """
    fig_generator 
    Genera una mappa interattiva delle sorgenti.
    WIP

    Args:
        catalog_path (str,pathlib.Path, optional): _description_. Defaults to custom_paths.csv_path.

        input_dataframe (pandas.DataFrame, optional): _description_. Defaults to None.

        html_output (str,pathlib.Path, optional): _description_. Defaults to custom_paths.htmlmap_path.

    Returns:
        ploty.express.fig: Canvas della mappa interattiva.
    """
    if input_dataframe is None:
        logger.info(f"Loading from {catalog_path}")
        dataframe = pd.read_csv(catalog_path)
    else:
        if catalog_path != custom_paths.csv_path:
            logger.warning(
                "'input_dataframe' detected, 'catalog_path' will be ignored.")
        logger.info("Loading DataFrame from input")
        dataframe = input_dataframe
    try:
        fig = px.scatter_geo(
            dataframe,
            lat="GLAT",
            lon="GLON",
            color="CLASS_GENERIC",
            hover_name="Source_Name",
            hover_data={
                "J2000_Name":True,
                "CLASS_GENERIC":True,
                "CLASS_DESCRIPTION": True,
                "GLAT":False,
                "GLON":False,
                },
            projection="mollweide",
            title="Sky Map",
            basemap_visible=False,
        )
    except ValueError:
        fig = px.scatter_geo(
            dataframe,
            lat="GLAT",
            lon="GLON",
            color="CLASS1",
            hover_name="Source_Name",
            hover_data={
                "GLON": False,
                "GLAT": False,
            },
            projection="mollweide",
            title="Sky Map",
            basemap_visible=False,
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
    logger.info(f"Mappa generata con successo! Aperta in: {html_output}")
    return fig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Genera la mappa in hmtl nel file di output."
    )
    parser.add_argument(
        "--input_path",
        "-i",
        default=f"{custom_paths.csv_path}",
        help="Path del cataologo in csv.",
    )
    parser.add_argument(
        "--html_path",
        "-o",
        default=f"{custom_paths.map_path}",
        help="Path di output del file html.",
    )
    args = parser.parse_args()

    fig_generator(
        catalog_path=args.input_path,
        html_output=args.html_path
    )

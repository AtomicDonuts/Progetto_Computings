"""
This module is responsible for generating interactive sky maps using Plotly.
It visualizes the galactic coordinates (GLAT, GLON) of sources and colors them
according to their class.
"""

import argparse

from loguru import logger
import plotly.express as px
import pandas as pd

# pylint: disable=import-error, wrong-import-position
from pathlib import Path
import sys

git_dir = None
for i in Path(__file__).parents:
    for j in i.iterdir():
        if ".git" in j.as_posix() and j.is_dir():
            git_dir = i
if git_dir is None:
    raise FileNotFoundError(
        "Git Directory Not Found. Please ensure that you cloned the repository in the right way."
    )
import_dir = git_dir / "imports/"
sys.path.append(import_dir.as_posix())
import custom_variables as custom_paths

# pylint: enable=import-error, wrong-import-position


def fig_generator(
    catalog_path=custom_paths.csv_path,
    input_dataframe=None,
    html_output=custom_paths.map_path,
):
    """
    Generates an interactive scatter_geo map of astronomical sources.

    The map uses Mollweide projection and displays custom hover information including
    source name, position, and classification. The result is saved as an HTML file.

    :param catalog_path: Path to the CSV catalog. Ignored if `input_dataframe` is provided.
                         Defaults to `custom_paths.csv_path`.
    :type catalog_path: str or pathlib.Path, optional
    :param input_dataframe: Pre-loaded Pandas DataFrame containing source data.
                            Defaults to None.
    :type input_dataframe: pandas.DataFrame, optional
    :param html_output: Output path for the HTML map file.
                        Defaults to `custom_paths.map_path`.
    :type html_output: str or pathlib.Path, optional
    :return: The generated Plotly figure object.
    :rtype: plotly.graph_objs.Figure
    """
    if input_dataframe is None:
        logger.info(f"Loading from {catalog_path}")
        dataframe = pd.read_csv(catalog_path)
    else:
        if catalog_path != custom_paths.csv_path:
            logger.warning(
                "'input_dataframe' detected, 'catalog_path' will be ignored."
            )
        logger.info("Loading DataFrame from input")
        dataframe = input_dataframe
    try:
        extra_cols = ["J2000_Name", "CLASS_TYPE", "CLASS_DESCRIPTION", "CLASS_DNN"]
        fig = px.scatter_geo(
            dataframe,
            lat="GLAT",
            lon="GLON",
            color="CLASS_GENERIC",
            hover_name="Source_Name",
            custom_data=extra_cols,
            projection="mollweide",
            title="Interactive Map",
            basemap_visible=False,
        )
        fig.update_traces(
            hovertemplate=(
                "<b>Source Name: %{hovertext}</b><br>"
                +
                "<i>Catalog Name: %{customdata[0]}</i><br>"
                +
                "---------------------<br>"
                +
                "GLAT: %{lat:.2f}°<br>"
                + 
                "GLON: %{lon:.2f}°<br>"
                +
                "Type: %{customdata[1]} (%{customdata[2]})<br>"
                +
                "DNN Prediction: %{customdata[3]}"
                +
                "<extra></extra>"
            )
        )
    except ValueError:
        logger.error("Something Went Wrong")
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
        fig.update_traces(
            hovertemplate=(
                "<b>Sorgente: %{hovertext}</b><br>"  # 'hover_name' popola automaticamente %{hovertext}
                + "---------------------<br>"
                + "<i>GLAT</i>: %{lat:.2f}°<br>"  # Formatta la latitudine a 2 decimali
                + "<i>GLON</i>: %{lon:.2f}°<br>"  # Formatta la longitudine a 2 decimali
                + "<extra></extra>"  # Rimuove il box laterale "trace 0"
            )
        )

    fig.update_geos(showframe=True, lataxis_showgrid=True, lonaxis_showgrid=True)

    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=1)))

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

    fig_generator(catalog_path=args.input_path, html_output=args.html_path)

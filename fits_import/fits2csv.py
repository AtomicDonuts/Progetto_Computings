"""
This module provides utilities to convert FITS catalog files into Pandas DataFrames and CSV files.
It handles parsing of FITS extensions, cleaning of array columns, and integration
of external DNN predictions into the dataset.
"""

import argparse

from astropy.io import fits
from astropy.table import Table
from loguru import logger
import pandas as pd
import numpy as np

# Import del modulo che contiene le variabili e i path della repository

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


def add_prediction(
    catalog_path=custom_paths.csv_path,
    input_dataframe=None,
    prediction_path=custom_paths.prediction_path,
):
    """
    Integrates the neural network predictions into the dataframe.

    It adds a 'CLASS_DNN' column to the dataframe based on the predictions stored in
    the numpy file.

    :param catalog_path: Path to the CSV catalog to load if input_dataframe is not provided.
                         Defaults to `custom_paths.csv_path`.
    :type catalog_path: str or pathlib.Path, optional
    :param input_dataframe: An existing DataFrame to modify. If provided, `catalog_path` is ignored.
    :type input_dataframe: pandas.DataFrame, optional
    :param prediction_path: Path to the numpy file (.npy) containing predictions.
                            Defaults to `custom_paths.prediction_path`.
    :type prediction_path: str or pathlib.Path, optional
    :return: The DataFrame with the added 'CLASS_DNN' column.
    :rtype: pandas.DataFrame
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

    prediction = np.load(prediction_path)
    dataframe["CLASS_DNN"] = np.where(
        (dataframe["CLASS_GENERIC"] == "No Association")
        | (dataframe["CLASS_GENERIC"] == "AGN")
        | (dataframe["CLASS_GENERIC"] == "Pulsar"),
        prediction,
        "Not Predicted",
    )
    return dataframe


def fits_to_pandas(fits_file_path=custom_paths.fits_path,
                   prediction_path=custom_paths.prediction_path,):
    """
    Parses a FITS file and converts it into a clean Pandas DataFrame.

    This function extracts data from the FITS extension, expands array columns (e.g., Flux_Band)
    into individual columns, normalizes string columns, and maps class codes to generic descriptions.
    It also optionally loads and merges DNN predictions.

    :param fits_file_path: Path to the input FITS catalog file.
                           Defaults to `custom_paths.fits_path`.
    :type fits_file_path: str or pathlib.Path
    :param prediction_path: Path to the prediction file to integrate.
                            Defaults to `custom_paths.prediction_path`.
    :type prediction_path: str or pathlib.Path
    :return: A cleaned and formatted DataFrame containing the catalog data.
    :rtype: pandas.DataFrame
    """
    logger.info(f"Parsing FITS: {fits_file_path}...")
    data = fits.getdata(fits_file_path, ext=1)
    data = Table(data)

    # Queste colonne contengono array di dati che possono servire
    # divido ogni array in un numero di colonne e poi le elimino
    col_name = [
        "Flux_Band",
        "nuFnu_Band",
        "Sqrt_TS_Band",
        "Flux_History",
        "Sqrt_TS_History",
    ]
    for col_array in col_name:
        for array_index in range(len(data[f"{col_array}"][0])):
            data[f"{col_array}_{array_index}"] = data[f"{col_array}"][:, array_index]

    data.remove_columns(
        [
            "Flux_Band",
            "Unc_Flux_Band",
            "nuFnu_Band",
            "Sqrt_TS_Band",
            "Flux_History",
            "Unc_Flux_History",
            "Sqrt_TS_History",
        ]
    )

    df = data.to_pandas()
    df[df.select_dtypes(["object"]).columns] = df.select_dtypes(["object"]).apply(
        lambda x: x.str.strip()
    )
    df["CLASS_TYPE"] = df["CLASS1"].str.upper()
    df["CLASS_DESCRIPTION"] = np.where(
        df["CLASS1"].str.isupper(), "Identified", "Associated"
    )
    df["CLASS_GENERIC"] = df["CLASS_TYPE"].replace(custom_paths.code_to_name)
    df["J2000_Name"] = df["Source_Name"]
    df["Source_Name"] = np.where(
        df["ASSOC1"] == "",
        df["Source_Name"],
        df["ASSOC1"],
    )
    df = df.drop(1389)
    if Path(prediction_path).exists():
        logger.info("DNN Predictions found.")
        df = add_prediction(input_dataframe=df,prediction_path = Path(prediction_path))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte il catalogo fits in un file csv."
    )
    parser.add_argument(
        "--input_path",
        "-i",
        default=f"{custom_paths.fits_path}",
        help="Path del cataologo fits.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=f"{custom_paths.csv_path}",
        help="Path di output del database csv.",
    )
    parser.add_argument(
        "--prediction_path",
        "-p",
        default=f"{custom_paths.csv_path}",
        help="Path delle prediction della rete neurale.",
    )
    args = parser.parse_args()
    dataf = fits_to_pandas(args.input_path,args.prediction_path)
    dataf.to_csv(args.output_path, index=False)
    logger.info(f"{Path(args.output_path).resolve()} saved.")

"""
Converte il catalogo fits in un database di pandas.
Se eseguito restituisce il catalogo convertito in csv nel Path
di output scelto.

Al momento restituisce un DataFrame per creare la mappa.
Le colonne scelte saranno poi aggiustate in futuro.
"""

import argparse

from astropy.io import fits
from astropy.table import Table
from loguru import logger
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

def fits_to_pandas(fits_file_path=custom_paths.fits_path):
    """
    fits_to_pandas
    Funzione che converte il catalogo in fits in un database Pandas
    Le informazioni esterne a "source" sono aggiunte in colonne
    apposite che iniziano 'spectrum_' e 'spatial_'.
    Args:
        fits_file_path (srt,pathlib.Path): Path del catalogo in formato .fits
        il default è definito del file "custom_variables.py"

    Returns:
        pandas.DataFrame: Pandas DataBase del Catalogo.
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
    args = parser.parse_args()
    fits_to_pandas(args.input_path).to_csv(args.output_path, index=False)
    logger.info(f"{Path(args.output_path).resolve()} saved.")

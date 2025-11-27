"""
Converte il catalogo fits in un database di pandas.
Se eseguito restituisce il catalogo convertito in csv nel Path
di output scelto.

Al momento restituisce un DataFrame per creare la mappa.
Le colonne scelte saranno poi aggiustate in futuro.
"""

import argparse
import sys

from astropy.io import fits
from astropy.table import Table
from loguru import logger
import numpy as np

# Import del modulo che contiene le variabili e i path della repository
sys.path.append("../imports/")
import custom_variables as custom_paths


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
    df["CLASS_GENERIC"] = df["CLASS1"].str.capitalize()
    df["CLASS_DESCRIPTION"] = np.where(
        df["CLASS1"].str.isupper(), "Identified", "Associated"
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
    df_data = fits_to_pandas(args.input_path).to_csv(args.output_path, index=False)
    logger.info(f"{custom_paths.Path(args.output_path).resolve()} saved.")

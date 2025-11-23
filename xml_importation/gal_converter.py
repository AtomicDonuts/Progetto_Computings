"""
Converte le coordinate nel database da equatoriali a galattiche.
"""
import sys

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from loguru import logger

sys.path.append("../imports/")
import custom_variables as custom_paths


def equatorial_to_galattic(
    catalog_path=custom_paths.csv_path,
    output_path=custom_paths.csv_path,
    keep_old_cord=False,
    dataframe_input = pd.DataFrame({}),
    save_as_file = True
):

    if not dataframe_input.empty:
        dataframe = dataframe_input

    else:
        try:
            dataframe = pd.read_csv(catalog_path)

        except FileNotFoundError:
            logger.error(
                f"Errore: Il file '{catalog_path}' non è stato trovato. Controlla il percorso."
            )
            return None

        except PermissionError:
            logger.error("Errore: Permesso negato.")
            return None

        except pd.errors.EmptyDataError:
            logger.error("Errore: Il file è vuoto (nessun dato o colonna).")
            return None

        except pd.errors.ParserError as exception:
            logger.error(
                f"Errore di parsing: Il formato del CSV non è valido. Dettagli: {exception}"
            )
            return None

        except UnicodeDecodeError:
            logger.error(
                "Errore di codifica: Prova a cambiare encoding (es. 'latin-1' o 'cp1252')."
            )
            return None

        except Exception as exception:
            logger.error(f"Si è verificato un errore imprevisto: {exception}")
            return None

    try:
        dataframe["RA_deg"] = dataframe["spatial_RA_value"].fillna(dataframe["RA"])
        dataframe["DEC_deg"] = dataframe["spatial_DEC_value"].fillna(dataframe["DEC"])

    except KeyError as exception:
        logger.error(
            f"KeyError: {exception}.\nProbabilmente il file è stato già convertito"\
            " o potresti aver aperto il file sbagliato.")
        return dataframe

    dataframe["RA_deg"]  = dataframe.apply(lambda row: row["RA_deg"]  * u.deg, axis=1)  
    dataframe["DEC_deg"] = dataframe.apply(lambda row: row["DEC_deg"] * u.deg, axis=1)
    c_galactic = SkyCoord(
        ra=dataframe["RA_deg"], dec=dataframe["DEC_deg"], frame="icrs"
    ).galattic
    dataframe["GLat"]  = c_galactic.l.degree  # type: ignore
    dataframe["GLong"] = c_galactic.b.degree # type: ignore

    if keep_old_cord:
        drop_col = ["RA_deg", "DEC_deg"]
    else:
        drop_col = [col for col in dataframe.columns if "RA" in col or "DEC" in col]
    
    dataframe = dataframe.drop(drop_col, axis=1)
    
    if save_as_file:
        dataframe.to_csv(output_path, index=False)
    else:
        return dataframe

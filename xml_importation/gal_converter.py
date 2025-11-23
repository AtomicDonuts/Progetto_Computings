"""
Converte le coordinate nel database da equatoriali a galattiche.
"""
import argparse
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
    input_dataframe = None,
    save_as_file = True,
    path_only_cord = custom_paths.gmap_path
):
    """
    equatorial_to_galattic
    Converte le coordinate equatoriali del dataframe in input in coordinate galattiche.
    Se non specificato, le colonne con le coordinate equatoriali sono poi rimosse.
    Di default il dataframe viene poi salvato in custom_path.csv_path oltre che ritornato
    dalla funzione.

    Args:
        catalog_path (str,pathlib.Path, optional): path del catalogo, viene ignorato se
        viene fornito un dataframe in input. Defaults to {custom_paths.csv_path}.

        output_path (str,pathlib.Path, optional): path del file di output. Defaults to
        {custom_paths.csv_path}.

        keep_old_cord (bool, optional): Ritorna il dataframe originale con appese due colonne
        contenenti le coordinate galattiche (LII,BII). Defaults to False.

        input_dataframe (pandas.DataFrame, optional): Se fornito, ignora il path del catalogo
        e utilizza il dataframe come input.

        save_as_file (bool, optional): Se True il dataframe generato viene salvato al
        path di output. Defaults to True.

        path_only_cord (str,pathlib.Path, optional): Salva nel path fornito un database
        che ha le colonne 'name','BII','LII'. Defaults to {custom_paths.gmap_path}.

    Returns:
        _type_: _description_
    """

    if input_dataframe is None:
        dataframe = pd.read_csv(catalog_path)
    else:
        if catalog_path != custom_paths.csv_path:
            logger.warning("'input_dataframe' detected, 'catalog_path' will be ignored.")
        dataframe = input_dataframe

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
    ).galactic
    dataframe["BII"]  = c_galactic.l.degree  # type: ignore
    dataframe["LII"] = c_galactic.b.degree # type: ignore

    if keep_old_cord:
        drop_col = ["RA_deg", "DEC_deg"]
    else:
        drop_col = [col for col in dataframe.columns if "RA" in col or "DEC" in col]

    if path_only_cord.name != '':
        dataframe[["name", "BII", "LII"]].to_csv(path_only_cord, index=False)

    dataframe = dataframe.drop(drop_col, axis=1)

    if save_as_file:
        dataframe.to_csv(output_path, index=False)
    return dataframe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte le coordinate equatoriali del dataframe in input"\
            "in coordinate galattiche. Se non specificato, le colonne con le " \
            "coordinate equatoriali sono poi rimosse. Di default il dataframe" \
            " viene poi salvato in custom_path.csv_path oltre che ritornato dalla funzione.")
    parser.add_argument(
        "--input_path",
        "-i",
        default=f"{custom_paths.csv_path}",
        help="Path del cataologo in csv.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=f"{custom_paths.csv_path}",
        help="Path di output del database convertito.",
    )
    parser.add_argument(
        "--maps_path",
        "-m",
        default=f"{custom_paths.gmap_path}",
        help="Path di output del database con solo il nome e le coordinate galattiche.",
    )
    args = parser.parse_args()
    equatorial_to_galattic(catalog_path=args.input_path,
                           output_path=args.output_path,
                           path_only_cord=custom_paths.Path(args.maps_path))
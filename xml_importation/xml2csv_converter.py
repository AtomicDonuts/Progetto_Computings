"""
Script che converte il catalogo xml in un database di pandas.
Se eseguito restituisce il catalogo convertito in csv nel Path
di output scelto
"""
import argparse
import sys
import pandas as pd
import xml.etree.ElementTree as ET
from loguru import logger

# Import del modulo che contiene le variabili e i path della repository
sys.path.append("../imports/")
import custom_variables as custom_paths


def xml_to_pandas(xml_file_path=custom_paths.xml_path):
    """
    xml_to_pandas
    Funzione che converte il catalogo in xml in un database Panda
    Le informazioni esterne a "source" sono aggiunte in colonne
    apposite che iniziano 'spectrum_' e 'spatial_'.
    Args:
        xml_file_path (_type_): Path del catalogo in formato .xml
        il default è definito del file "custom_variables.py"

    Returns:
        Pandas DataBase del Catalogo.
    """
    logger.info(f"Parsing XML: {xml_file_path}...")
    # Con queste parole si escludono delle colonne che sono costanti
    exclude_words = ["name", "free", "min", "max", "scale"]
    all_sources = []

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Errore parsing XML: {e}")
        return pd.DataFrame(all_sources)

    for source in root.findall("source"):
        source_data = {}

        source_data.update(source.attrib)

        spectrum = source.find("spectrum")
        if spectrum is not None:
            source_data["spectrum_type"] = spectrum.get("type")
            for param in spectrum.findall("parameter"):
                p_name = param.get("name")
                for key, value in param.attrib.items():
                    if key not in exclude_words:
                        col_name = f"spectrum_{p_name}_{key}"
                        source_data[col_name] = value

        spatial = source.find("spatialModel")
        if spatial is not None:
            source_data["spatialModel_type"] = spatial.get("type")
            for param in spatial.findall("parameter"):
                p_name = param.get("name")
                for key, value in param.attrib.items():
                    if key not in exclude_words:
                        col_name = f"spatial_{p_name}_{key}"
                        source_data[col_name] = value

        all_sources.append(source_data)

    df = pd.DataFrame(all_sources)
    # Dopo aver creato il database, convertiamo le colonne numeriche in float
    # Dato che l'opzione db.apply(fun,errors = "ignore") è deprecata, escludiamo
    # le colonne i cui argomenti sono stringe e non floats e poi applichiamo globalmente
    # al resto del database la funzione ".to_numeric" che converte le stringe in floats
    exclude_col = ["name", "type", "spectrum_type", "spatialModel_type"]
    col = df.columns.drop(exclude_col)
    df[col] = df[col].apply(pd.to_numeric)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
             description="Script che converte il catalogo xml in un file csv.")
    parser.add_argument("--input_path",
                        "-i",
                        default=f"{custom_paths.xml_path}",
                        help="Path del cataologo in xml.")
    parser.add_argument("--output_path",
                        "-o",
                        default=f"{custom_paths.csv_path}",
                        help="Path di output del database in csv.")
    args = parser.parse_args()
    df_data = xml_to_pandas(args.input_path).to_csv(args.output_path)
    logger.info(f"{custom_paths.Path(args.output_path).resolve()} saved.")

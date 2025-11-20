import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import ROOT
from array import array


def xml_to_pandas(xml_file_path):
    """ """
    exclude_words = ["name", "free", "min", "max", "scale"]
    print(f"Parsing XML: {xml_file_path}...")
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Errore parsing XML: {e}")
        return None

    all_sources = []

    # 1. Parsing e Appiattimento (Flattening)
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
    exclude_col = ["name", "type", "spectrum_type", "spatialModel_type"]
    col = df.columns.drop(exclude_col)
    df[col] = df[col].apply(pd.to_numeric)
    return df


file_xml = "./gll_psc_v32.xml"


df_data = xml_to_pandas(file_xml).to_csv("gll_psc_v32.csv")

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


def save_to_root_classic(df, output_filename):
    """
    Converte il DataFrame in TTree.
    """
    print(f"Creazione file ROOT: {output_filename}")

    tfile = ROOT.TFile(output_filename, "RECREATE")
    tree = ROOT.TTree("FermiCatalog", "Catalogo sorgenti Fermi-LAT")

    # Dizionari per mantenere i puntatori ai buffer per C++
    buffers = {}

    print("Configurazione dei Branch...")

    for col in df.columns:
        numeric_series = pd.to_numeric(df[col], errors="coerce")

        if not numeric_series.isna().all():
            # Tipo: Numerico (Float/Double)
            df[col] = numeric_series.fillna(np.nan)
            buffers[col] = array("d", [0.0])
            tree.Branch(col, buffers[col], f"{col}/D")
        else:
            # Tipo: Stringa (Testo)
            df[col] = df[col].astype(str).fillna("")
            buffers[col] = ROOT.std.string("")
            tree.Branch(col, buffers[col])

    # --- Riempimento del Tree ---
    print(f"Riempimento del Tree con {len(df)} sorgenti...")

    for row in df.itertuples(index=False):
        for col_name in df.columns:
            val = getattr(row, col_name)

            if isinstance(buffers[col_name], ROOT.std.string):
                # Assegnazione stringa C++
                buffers[col_name].assign(str(val))
            else:
                # Assegnazione numero
                buffers[col_name][0] = float(val)

        tree.Fill()

    # --- Salvataggio ---
    tfile.Write()
    tfile.Close()
    print("File salvato e chiuso correttamente.")

file_xml = "gll_psc_v32.xml"
file_root = "gll_psc_v32.root"


# 1. XML -> Pandas
df_data = xml_to_pandas(file_xml)

if df_data is not None:
    # 2. Pandas -> ROOT (Metodo Classico TTree)
    save_to_root_classic(df_data, file_root)

    # Verifica rapida
    print("\nVerifica lettura file:")
    f = ROOT.TFile.Open(file_root)
    t = f.Get("FermiCatalog")
    print(f"Sorgenti salvate nel TTree: {t.GetEntries()}")

df_data.to_csv("gll_psc_v32.csv")

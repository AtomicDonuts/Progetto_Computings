"""
Definizione di alcune variabili globali.
Deve essere importato nei vari script utilizzando la seguente sintassi:
import sys
sys.path.append("../imports/")
import custom_variables as custom_paths
"""

from pathlib import Path

# Folders
if Path("../imports").exists():
    dir_imports_path = Path("../imports")
else:
    raise FileNotFoundError("'../imports' not found.")

if Path("../map").exists():
    dir_map_path = Path("../map")
else:
    raise FileNotFoundError("'../map' not found.")

if Path("../files").exists():
    dir_files_path = Path("../files")
else:
    raise FileNotFoundError("'../files' not found.")

if Path("../fits_import").exists():
    dir_files_path = Path("../fits_import")
else:
    raise FileNotFoundError("'../fits_import' not found.")

if Path("../ann").exists():
    dir_files_path = Path("../ann")
else:
    raise FileNotFoundError("'../ann' not found.")

# Files
fits_path = Path("../files/gll_psc_v35.fit")
csv_path = Path("../files/gll_psc.csv")
gmap_path = Path("../files/galattic_coordinates.csv")
map_path = Path("../map/index.html")
# docs_path = Path("../docs/index.html")

# Dictionaries
name_to_code = {
    "Pulsar": ["PSR"],
    "AGN": ["BLL", "FSRQ", "RDG", "AGN", "SSRQ", "CSS", "BCU", "NLSY1", "SEY"],
    "Globular cluster": "glc",
    "Starburst galaxy": "sbg",
    "PWN": "pwn",
    "SNR": "snr",
    "SNR or PWN": "spp",
    "Binary": ["bin", "hmb", "lmb"],
    "Galaxy": "gal",
    "Nova": "nov",
    "Star-Forming Regions": "sfr",
    "Uknown": "unk",
    "No Association": "",
}
code_to_name = {
    "PSR": "Pulsar",
    "MSP": "Pulsar",
    "AGN": "AGN",
    "BLL": "AGN",
    "FSRQ": "AGN",
    "RDG": "AGN",
    "SSRQ": "AGN",
    "CSS": "AGN",
    "BCU": "AGN",
    "NLSY1": "AGN",
    "SEY": "AGN",
    "GLC": "Globular cluster",
    "SBG": "Starburst galaxy",
    "PWN": "PWN",
    "SNR": "SNR",
    "SPP": "SNR or PWN",
    "BIN": "Binary",
    "HMB": "Binary",
    "LMB": "Binary",
    "GAL": "Galaxy",
    "NOV": "Nova",
    "SFR": "Star-Forming Regions",
    "GC": "Galactic Center",
    "UNK": "Unknown",
    "": "No Association",
}

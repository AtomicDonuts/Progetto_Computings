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
if Path("../xml_import").exists():
    dir_xml_path = Path("../xml_importation")
else:
    raise FileNotFoundError("'../xml_importation' not found.")
if Path("../files").exists():
    dir_files_path = Path("../files")
else:
    raise FileNotFoundError("'../files' not found.")

# Files
xml_path = Path("../files/gll_psc_v32.xml")
fits_path = Path("../files/gll_psc_v35.fit")
csv_path = Path("../files/gll_psc.csv")
gmap_path = Path("../files/galattic_coordinates.csv")
htmlmap_path = Path("../files/galactic_map.html")
docs_path = Path("../docs/index.html")
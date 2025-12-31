"""
This module defines global variables and file paths used throughout the project.
It dynamically locates the git root directory to ensure paths work regardless of
execution context.
"""

from pathlib import Path

GIT_DIR = None
for i in Path(__file__).parents:
    for j in i.iterdir():
        if ".git" in j.as_posix() and j.is_dir():
            GIT_DIR = i
if GIT_DIR is None:
    raise FileNotFoundError(
        "Git Directory Not Found. Please ensure that you cloned the repository in the right way."
    )

# Folders

dir_imports_path = Path(GIT_DIR / "imports")
dir_map_path = Path(GIT_DIR / "map")
dir_files_path = Path(GIT_DIR / "files")
dir_fits_import_path = Path(GIT_DIR / "fits_import")
dir_ann_path = Path(GIT_DIR / "dnn")
dir_models_path = Path(dir_ann_path / "keras_models")
dir_tuner_path = Path(dir_ann_path / "tuner")

# Files
fits_path = dir_files_path / "gll_psc_v35.fit"
csv_path = dir_files_path / "gll_psc.csv"
gmap_path = dir_files_path / "galattic_coordinates.csv"
map_path = dir_map_path / "index.html"
dnnmap_path = dir_map_path / "index2.html"

model_path = dir_models_path / "prediction_model.keras"
prediction_path = dir_models_path / "prediction.npy"
png_path = dir_models_path / "model.png"

train_data_path = dir_tuner_path / "train_data.npy"
train_label_path = dir_tuner_path / "train_label.npy"
validation_data_path = dir_tuner_path /  "validation_data.npy"
validation_label_path = dir_tuner_path / "validation_label.npy"

# Dictionaries
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

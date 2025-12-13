"""
This module defines global variables and file paths used throughout the project.
It dynamically locates the git root directory to ensure paths work regardless of
execution context.
"""

from pathlib import Path

git_dir = None
for i in Path(__file__).parents:
    for j in i.iterdir():
        if ".git" in j.as_posix() and j.is_dir():
            git_dir = i
if git_dir is None:
    raise FileNotFoundError(
        "Git Directory Not Found. Please ensure that you cloned the repository in the right way."
    )

# Folders

if Path(git_dir / "imports").exists():
    dir_imports_path = Path(git_dir / "imports")
else:
    raise FileNotFoundError(f"{Path(git_dir / 'imports')} not found.")

if Path(git_dir / "map").exists():
    dir_map_path = Path(git_dir / "map")
else:
    raise FileNotFoundError(f"{Path(git_dir / 'map')} not found.")

if Path(git_dir / "files").exists():
    dir_files_path = Path(git_dir / "files")
else:
    raise FileNotFoundError(f"{Path(git_dir / 'files')} not found.")

if Path(git_dir / "fits_import").exists():
    dir_fits_import_path = Path(git_dir / "fits_import")
else:
    raise FileNotFoundError(f"{Path(git_dir / 'fits_import')} not found.")

if Path(git_dir / "dnn").exists():
    dir_ann_path = Path(git_dir / "dnn")
else:
    raise FileNotFoundError(f"{Path(git_dir / 'dnn')} not found.")

if Path(dir_ann_path / "keras_models").exists():
    dir_models_path = Path(dir_ann_path / "keras_models")
else:
    raise FileNotFoundError(f"{Path(dir_ann_path / 'keras_models')} not found.")

#   Folder Import Dinamica
#   VSCode lo odia partiolarmente tanto

# required_dirs = [
#     dir.name for dir in git_dir.iterdir() if (dir.is_dir()) and ("." not in dir.name)
# ]

# for folder in required_dirs:
#     target_path = git_dir / folder
#     if target_path.exists():
#         globals()[f"dir_{folder}_path"] = target_path
#     else:
#         raise FileNotFoundError(f"{target_path} not found.")


# Files
fits_path = dir_files_path / "gll_psc_v35.fit"
csv_path = dir_files_path / "gll_psc.csv"
gmap_path = dir_files_path / "galattic_coordinates.csv"
map_path = dir_map_path / "index.html"

model_path = dir_models_path / "prediction_model.keras"
prediction_path = dir_models_path / "prediction.npy"
png_path = dir_models_path / "model.png"

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

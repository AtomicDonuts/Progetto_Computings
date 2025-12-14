"""
Script entry point for the GitHub Action map generation workflow.
It orchestrates the conversion of FITS files to CSV and the subsequent generation
of the HTML map.
"""

# pylint: disable=import-error , wrong-import-position
from pathlib import Path
import sys

git_dir = None
for i in Path(__file__).parents:
    for j in i.iterdir():
        if ".git" in j.as_posix() and j.is_dir():
            git_dir = i
if git_dir is None:
    raise FileNotFoundError(
        "Git Directory Not Found. Please ensure that you cloned the repository in the right way."
    )
import_dir = git_dir / "imports/"
sys.path.append(import_dir.as_posix())

import custom_variables as custom_paths
sys.path.append(custom_paths.dir_fits_import_path.as_posix())
sys.path.append(custom_paths.dir_map_path.as_posix())
import fits2csv as fit_fun
import map_example as map_fun

# pylint: enable=import-error, wrong-import-position

map_fun.fig_generator(
    input_dataframe=fit_fun.fits_to_pandas(),
    html_output=custom_paths.map_path,
)
map_fun.fig_generator_dnn(
    input_dataframe=fit_fun.fits_to_pandas(),
    html_output=custom_paths.dnnmap_path,
)
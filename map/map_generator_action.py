"""
Questo modulo genera la mappa a partire da file FITS, per la GitHub Action
"""
import sys
sys.path.append("../imports/")
sys.path.append("../fits_import/")

# pylint: disable=import-error
# pylint: disable=wrong-import-position
import custom_variables as custom_paths
import fits2csv as fit_fun
import map_example as map_fun

# pylint: enable=import-error, wrong-import-position

map_fun.fig_generator(
    input_dataframe=fit_fun.fits_to_pandas(),
    html_output=custom_paths.map_path,
)
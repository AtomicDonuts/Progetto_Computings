# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../imports"))
sys.path.insert(0, os.path.abspath("../../dnn"))
sys.path.insert(0, os.path.abspath("../../map"))
sys.path.insert(0, os.path.abspath("../../fits_import"))

project = 'DNN-4FGL'
copyright = '2025, Pasquale Napoli'
author = 'Pasquale Napoli'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Legge le docstring dal codice
    "sphinx.ext.viewcode",  # Aggiunge link al codice sorgente
    "sphinx.ext.napoleon",  # Supporta stile Google/NumPy (opzionale, ma utile)
    "sphinx.ext.mathjax",  # Per formule matematiche se ne usi
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ['_static']

autodoc_mock_imports = [
    "tensorflow",
    "keras",
    "numpy",
    "pandas",
    "matplotlib",
    "plotly",
    "astropy",
    "sklearn",
]

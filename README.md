# Fermi-LAT Source Classification

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/atomicdonuts/progetto_computings/map_generator.yml?label=Map%20Generator&logo=github-actions)
![Python Version](https://img.shields.io/badge/python-3.10-blue?logo=python)

Project developed by **Pascal Napoli** for the **Computing Methods For Experimental Physics And Data Analysis** exam.

## 🔭 Project Description

This project implements a data analysis pipeline for the automatic classification of astrophysical sources from the **Fermi-LAT 4FGL catalog**.
The core of the work is an **Artificial Neural Network (ANN)** designed to analyze the spectral characteristics of the sources and classify them into their respective astrophysical categories.

## 🌍 Interactive Map

An automatically generated interactive map displaying the distribution of the classified sources is available.

👉 **[Click here to view the Interactive Map](https://atomicdonuts.github.io/Progetto_Computings/map/)**

## 📂 Repository Structure

* `ann/`: Jupyter Notebooks and scripts for the Neural Network.
* `map/`: Scripts for generating the interactive map.
* `fits_import/`: Modules for processing FITS files.
* `.github/workflows/`: Workflows for automation.

## ⚙️ Repository Management

The project follows structured development practices:
* **CI/CD**: Uses **GitHub Actions** for the automatic generation and deployment of the updated map to GitHub Pages.
* **Issues**: Tracks development and bugs via GitHub Issues for organized workflow management.


## 📦 Installation

All necessary dependencies (e.g., `astropy`, `tensorflow`, `plotly`) are listed in the `requirements.txt` file.

To install the environment:

```bash
pip install -r requirements.txt
```
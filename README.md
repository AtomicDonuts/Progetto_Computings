# Fermi-LAT Source Classification

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/atomicdonuts/progetto_computings/gh-page.yml?label=Map%20Generator&logo=github-actions)
![Python Version](https://img.shields.io/badge/python-3.10-blue?logo=python)

Project developed by **Pascal Napoli** for the **Computing Methods For Experimental Physics And Data Analysis** exam.

## 📝 Project Description

This project implements a complete data analysis pipeline for the automatic classification of astrophysical sources from the **Fermi-LAT 4FGL catalog**. The primary goal is to distinguish between the two largest classes of gamma-ray emitters—**Active Galactic Nuclei (AGN)** and **Pulsars**—and to predict classifications for unassociated sources based on their spatial, temporal, and spectral characteristics.

The core of the project is a **Deep Neural Network (DNN)** trained to analyze these features with high accuracy.

## 🗺️ Interactive Map

An automatically generated interactive map displaying the distribution of classified sources is available.
* **Visualization:** Uses **Plotly** to project sources onto a **Mollweide projection** using Galactic Coordinates.
* **Automation:** A GitHub Action automatically regenerates and deploys the map to GitHub Pages whenever the dataset or map script is updated.

👉 **[Click here to view the Interactive Map](https://atomicdonuts.github.io/Progetto_Computings/map/)**

## 📘 Documentation

Complete documentation for the project is available here:
👉 **[Read the Documentation](https://atomicdonuts.github.io/Progetto_Computings/docs/)**

## 📂 Repository Structure

* `dnn/`: Contains Jupyter Notebooks, training scripts, and the model architecture.
* `map/`: Scripts for generating the interactive HTML map.
* `fits_import/`: Modules for processing raw FITS files into CSV format.
* `imports/`: Custom utility modules used across the project.
* `files/`: Contains the raw dataset files.
* `.github/workflows/`: CI/CD workflows for automation.

## 🛠️ Installation

All necessary dependencies (e.g., `astropy`, `tensorflow`, `plotly`, `pandas`) are listed in the `requirements.txt` file.

To install the environment:

```bash
pip install -r requirements.txt
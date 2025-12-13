"""
docstring
"""

import argparse
import sys
from pathlib import Path

# pylint: disable=import-error, wrong-import-position
from loguru import logger
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler

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

# pylint: enable=import-error, wrong-import-position

def model_prediction(
    catalog_path=custom_paths.csv_path,
    model_path=custom_paths.model_path,
    threshold = 0.63
):
    '''
    model_prediction _summary_

    Args:
        catalog_path (_type_, optional): _description_. Defaults to custom_paths.csv_path.
        model_path (_type_, optional): _description_. Defaults to custom_paths.model_path.
        prediction_path (_type_, optional): _description_. Defaults to custom_paths.prediction_path.
        threshold (float, optional): _description_. Defaults to 0.63.
    '''
    logger.info("Importing Model..")
    model = keras.models.load_model(model_path)
    logger.info("Importing Catalog")
    df = pd.read_csv(catalog_path)

    df["PowerLaw"] = np.where(df["SpectrumType"] == "PowerLaw",1,0,)
    df["LogParabola"] = np.where(df["SpectrumType"] == "LogParabola",1,0,)
    df["PLSuperExpCutoff"] = np.where(df["SpectrumType"] == "PLSuperExpCutoff",1,0,)

    col_input1 = [
        "GLAT",
        "Variability_Index",
        "PowerLaw",
        "LogParabola",
        "PLSuperExpCutoff",
    ]

    col_flux_band = np.array([[f"Flux_Band_{i}", f"Sqrt_TS_Band_{i}"] for i in range(8)])
    col_flux_hist = np.array([[f"Flux_History_{i}", f"Sqrt_TS_History_{i}"] for i in range(14)])

    norm_cols = np.array(list(col_flux_band.flatten()) + list(col_flux_hist.flatten()))
    scaler = StandardScaler()
    scaler.fit(df[norm_cols])
    scaled_data = scaler.transform(df[norm_cols])
    df[norm_cols] = scaled_data

    input_additional = df[col_input1].to_numpy()
    input_flux_band = df[col_flux_band.flatten()].to_numpy()
    input_flux_hist = df[col_flux_hist.flatten()].to_numpy()

    logger.info("Starting Predictions...")
    predictions = model.predict([input_flux_band, input_flux_hist, input_additional])
    th_pred = (predictions >= threshold).astype(int)
    th_pred = np.where(
        th_pred == 0,
        "AGN",
        "Pulsar",
    )
    th_pred = th_pred.reshape(len(th_pred))
    return th_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crea l'array delle predizioni per la rete neurale."
    )
    parser.add_argument(
        "--csv_path",
        "-i",
        default=f"{custom_paths.csv_path}",
        help="Path del cataologo csv.",
    )
    parser.add_argument(
        "--threshold",
        "-th",
        default=0.63,
        help="Threshold per dividere AGN da Pulsar.",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        default=f"{custom_paths.model_path}",
        help="Path del modello della rete neurale già allenato.",
    )
    parser.add_argument(
        "--prediction_path",
        "-p",
        default=f"{custom_paths.prediction_path}",
        help="Path delle prediction della rete neurale.",
    )
    args = parser.parse_args()
    preds = model_prediction(
    catalog_path= args.csv_path,
    model_path=args.model_path,
    threshold = args.threshold
    )
    logger.info("Saving Predictions..")
    np.save(args.prediction_path,preds)

"""
This script performs the training of the Deep Neural Network (DNN) using a multi-input architecture.

It handles data loading, feature engineering, normalization, class balancing,
and hyperparameter tuning using Keras Tuner (Hyperband).
The training process utilizes Stratified K-Fold cross-validation to ensure model robustness.
Results, including accuracy, F1 score, and confusion matrices, are logged and best models are saved.
"""

import os

# Imposta la variabile per nascondere la GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
from pathlib import Path

# pylint: disable=import-error, wrong-import-position
from loguru import logger
import numpy as np
import pandas as pd
import keras
from keras.models import clone_model
import keras_tuner as kt
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # pylint: disable=no-name-in-module


GIT_DIR = None
for i in Path(__file__).parents:
    for j in i.iterdir():
        if ".git" in j.as_posix() and j.is_dir():
            GIT_DIR = i
if GIT_DIR is None:
    raise FileNotFoundError(
        "Git Directory Not Found. Please ensure that you cloned the repository in the right way."
    )
import_dir = GIT_DIR / "imports/"
sys.path.append(import_dir.as_posix())
import nn_models as ann
import custom_variables as custom_paths
import metrics as met

# pylint: enable=import-error, wrong-import-position

logger.debug("Loading Catalog..")
df = pd.read_csv(custom_paths.csv_path)
df = df[(df["CLASS_GENERIC"] == "AGN") | (df["CLASS_GENERIC"] == "Pulsar")]
logger.debug(f"Sample Size: {len(df)}")

col_input1 = ["GLAT","Variability_Index","PowerLaw","LogParabola","PLSuperExpCutoff",]

col_flux_band = np.array([[f"Flux_Band_{i}", f"Sqrt_TS_Band_{i}"] for i in range(8)])
col_flux_hist = np.array([[f"Flux_History_{i}", f"Sqrt_TS_History_{i}"] for i in range(14)])

logger.debug("Normalizing Columns..")

norm_cols = np.array(list(col_flux_band.flatten()) + list(col_flux_hist.flatten()))
scaler = StandardScaler()
scaler.fit(df[norm_cols])
scaled_data = scaler.transform(df[norm_cols])
df[norm_cols] = scaled_data


input_additional = df[col_input1].to_numpy()
input_flux_band = df[col_flux_band.flatten()].to_numpy()
input_flux_hist = df[col_flux_hist.flatten()].to_numpy()
logger.info(f"Additionl Size: {input_additional.shape}")
logger.info(f"Flux_Band Size: {input_flux_band.shape}")
logger.info(f"Flux_History Size: {input_flux_hist.shape}")

logger.debug("Creating Labels..")

is_agn = df["CLASS_GENERIC"].to_numpy() == "AGN"

labels = np.zeros((len(df)), dtype=int)
labels[~is_agn] = 1

logger.debug("Creating Class Weights..")
class_weight = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(labels), y=labels
)
class_weight = {index: value for index, value in enumerate(class_weight)}

logger.debug("Splitting Dataset in Train e Test..")
splitdata = StratifiedKFold(n_splits=4, shuffle=True)
train, test = next(splitdata.split(np.zeros(len(labels)), labels))

fb = input_flux_band[train]
hb = input_flux_hist[train]
ia = input_additional[train]
lab = labels[train]
vfb = input_flux_band[test]
vhb = input_flux_hist[test]
via = input_additional[test]
vlab = labels[test]

logger.debug("Start Tuner...")
PROJECT_NAME = "TripleInput"
tuner = kt.Hyperband(
    ann.hp_final_model,
    objective="val_loss",
    max_epochs=100,
    factor=3,
    overwrite=False,
    directory=custom_paths.dir_tuner_path,
    project_name=PROJECT_NAME,
)
stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
tuner.search(
    x=[fb, hb, ia],
    y=lab,
    epochs=50,
    validation_split=0.5,
    class_weight=class_weight,
    callbacks=[stop_early],
    verbose=2,
)
logger.debug("Tuner Finished")

best_model = tuner.get_best_models(num_models=1)[0]
best_lr = best_model.optimizer.learning_rate

loss_array = []
auc_array = []
accuracy_array = []
acc_agn_array = []
acc_psr_array = []
eq_th_array = []
eq_acc_agn_array = []
eq_acc_psr_array = []
f1_array = []
th_array = []
cm_array = []

logger.debug("Starting Traning for the best model with KFold")
FOLD_NO = 0
skf = StratifiedKFold(n_splits=10, shuffle=True)
for ktrain, ktest in skf.split(np.zeros(len(lab)), lab):
    k_hb = hb[ktrain]
    k_fb = fb[ktrain]
    k_ia = ia[ktrain]
    k_lab = lab[ktrain]
    k_vfb = fb[ktest]
    k_vhb = hb[ktest]
    k_via = ia[ktest]
    k_vlab = lab[ktest]

    logger.info(f"Fold No.{FOLD_NO}")

    reset_model = clone_model(best_model)
    reset_model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            "auc",
        ],
    )
    reset_model.optimizer.learning_rate = best_lr

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

    logger.info("Training...")
    history = reset_model.fit(
        x=[k_fb, k_hb, k_ia],
        y=k_lab,
        epochs=300,
        validation_data=[[k_vfb, k_vhb, k_via], k_vlab],
        callbacks=[early_stopping, reduce_lr],
        verbose=0,  # pyright: ignore[reportArgumentType]
    )
    logger.info(
        "------------------------------------------------------------------------"
    )

    logger.info("Prediction on Evaluation Dataset")
    scores = reset_model.evaluate(
        [vfb, vhb, via],
        vlab,
        verbose=0,  # pyright: ignore[reportArgumentType]
    )
    predictions = reset_model.predict(
        [vfb, vhb, via],
        verbose=0,  # pyright: ignore[reportArgumentType]
    )

    loss_array.append(scores[0])
    auc_array.append(scores[1])
    logger.info(f"Loss: {scores[0]:.5F}")
    logger.info(f"AUC {(scores[2]*100):.2F}")

    acc, th = met.best_accuracy(vlab, predictions)
    accuracy_array.append(acc)
    th_array.append(th)
    logger.info(f"Accuracy: {(acc*100):.2F}")

    f1_score = met.f1_score(th, vlab, predictions)
    f1_array.append(f1_score)
    logger.info(f"F1 Score: {(f1_score*100):.2F}")

    eq_acc_agn, eq_acc_psr, eq_th = met.best_eq_accuracy(vlab, predictions)
    eq_th_array.append(eq_th)
    eq_acc_agn_array.append(eq_acc_agn)
    eq_acc_psr_array.append(eq_acc_psr)
    logger.info(f"EqAcc AGN: {(eq_acc_agn*100):.2F}")
    logger.info(f"EqAcc PSR: {(eq_acc_psr*100):.2F}")

    acc_agn, acc_psr = met.class_accuracy(th, vlab, predictions)
    acc_agn_array.append(acc_agn)
    acc_psr_array.append(acc_psr)
    logger.info(f"Accuracy AGN: {(acc_agn*100):.2F} Accuracy PSR: {(acc_psr*100):.2F}")

    th_pred = (predictions >= th).astype(int)
    cm_sing = met.sk_metrics.confusion_matrix(vlab, th_pred)
    logger.info(f"Confusion Matrix:\n{cm_sing}")
    cm_array.append(cm_sing)
    logger.info(
        "------------------------------------------------------------------------"
    )

    modelpath = custom_paths.dir_models_path / f"{PROJECT_NAME}_{FOLD_NO}.keras"
    reset_model.save(modelpath)

    FOLD_NO = FOLD_NO + 1
# end for
cm_array = np.array(cm_array)
logger.debug("Training End.")
logger.info("------------------------------------------------------------------------")
logger.info(f"Best Model Was: {np.argmax(f1_array)} (Based on F1Score)")
FORMATTED_TH = ", ".join([f"{x:.2f}" for x in th_array])
logger.info(f"Tresholds: [{FORMATTED_TH}]")
logger.info(f"Tresholds used: {th_array[np.argmax(f1_array)]}")
logger.info("------------------------------------------------------------------------")
logger.info(f"Dense Layer: {best_model.layers[3].units}")
logger.info(f"Dropout Rate: {best_model.layers[6].rate}")
logger.info("------------------------------------------------------------------------")
logger.info("Prediction on Evaluation DataSet")
logger.info(f"> Loss: {np.mean(loss_array):.5F}(+- {np.std(loss_array):.5F})")
logger.info(f"> AUC: {(np.mean(auc_array)*100):.2F} (+- {(np.std(auc_array)*100):.2F})")
logger.info(
    f"> Accuracy: {(np.mean(accuracy_array)*100):.2F} (+- {(np.std(accuracy_array)*100):.2F})"
)
logger.info(f"> F1: {(np.mean(f1_array)*100):.2F} (+- {(np.std(f1_array)*100):.2F})")
logger.info(
    f"> EqAcc AGN: {(np.mean(eq_acc_agn_array)*100):.2F} (+- {(np.std(eq_acc_agn_array)*100):.2F})"
)
logger.info(
    f"> EqAcc PSR: {(np.mean(eq_acc_psr_array)*100):.2F} (+- {(np.std(eq_acc_psr_array)*100):.2F})"
)
logger.info(
    f"> Acc AGN: {(np.mean(acc_agn_array)*100):.2F} (+- {(np.std(acc_agn_array)*100):.2F})"
)
logger.info(
    f"> Acc PSR: {(np.mean(acc_psr_array)*100):.2F} (+- {(np.std(acc_psr_array)*100):.2F})"
)
logger.info("Confution Matrix: ")
logger.info(
    f"\t{np.mean(cm_array[:,0,0])}+-{np.std(cm_array[:,0,0]):.2F}"
    f"\t{np.mean(cm_array[:,0,1])}+-{np.std(cm_array[:,0,1]):.2F}"
)
logger.info(
    f"\t{np.mean(cm_array[:,1,0])}+-{np.std(cm_array[:,1,0]):.2F}"
    f"\t{np.mean(cm_array[:,1,1])}+-{np.std(cm_array[:,1,1]):.2F}"
)
logger.info("------------------------------------------------------------------------")

"""
This script performs the training of the Deep Neural Network (DNN) using a single-input architecture.

It handles data loading, feature engineering, normalization, class balancing,
and hyperparameter tuning using Keras Tuner (Hyperband).
The training process utilizes Stratified K-Fold cross-validation to ensure model robustness.
Results, including accuracy, F1 score, and confusion matrices, are logged and best models are saved.
"""

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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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

logger.debug("Loading Catalog...")
df = pd.read_csv(custom_paths.csv_path)
df = df[(df["CLASS_GENERIC"] == "AGN") | (df["CLASS_GENERIC"] == "Pulsar")]
logger.debug(f"Sample Size: {len(df)}")

input_col = [
    "GLON",
    "GLAT",
    "Variability_Index",
    "PowerLaw",
    "LogParabola",
    "PLSuperExpCutoff",
]

inputs_data = df[input_col].to_numpy()
logger.debug(f"Additionl Size: {inputs_data.shape}")

logger.debug("Creating Labels...")
is_agn = df["CLASS_GENERIC"].to_numpy() == "AGN"
labels = np.zeros((len(df)), dtype=int)
labels[~is_agn] = 1

logger.debug("Creating Class Weights...")
class_weight = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(labels), y=labels
)
class_weight = {index: value for index, value in enumerate(class_weight)}

logger.debug("Splitting Dataset in Train e Test...")
splitdata = StratifiedKFold(n_splits=4, shuffle=True)
train, test = next(splitdata.split(np.zeros(len(labels)), labels))

train_data = inputs_data[train]
train_labels = labels[train]

validation_data = inputs_data[test]
validation_labels = labels[test]

logger.debug("Start Tuner...")
tuner = kt.Hyperband(
    ann.hp_final_model,
    objective="val_loss",
    max_epochs=100,
    factor=3,
    overwrite=False,
    directory="Progetto-GPU",
    project_name="Single-Input",
)
stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
tuner.search(
    x=train_data,
    y=train_labels,
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
for ktrain, ktest in skf.split(np.zeros(len(train_labels)), train_labels):
    kfold_train_data = train_data[ktrain]
    kfold_train_labels = train_labels[ktrain]
    kfold_validation_data = train_data[ktest]
    kfold_validation_labels = train_labels[ktest]

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

    history = reset_model.fit(
        x=kfold_train_data,
        y=kfold_train_labels,
        epochs=300,
        validation_data=[kfold_validation_data, kfold_validation_labels],
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight,
        verbose=2,  # pyright: ignore[reportArgumentType]
    )

    print(f"Fold No.{FOLD_NO}")
    print("------------------------------------------------------------------------")

    print("Prediction on Evaluation Dataset")
    scores = reset_model.evaluate(
        validation_data,
        validation_labels,
        verbose=2,  # pyright: ignore[reportArgumentType]
    )
    predictions = reset_model.predict(
        validation_data,
        verbose=2,  # pyright: ignore[reportArgumentType]
    )  

    loss_array.append(scores[0])
    auc_array.append(scores[1])
    print(f"Loss: {scores[0]}")
    print(f"AUC {scores[2]}")

    acc, th = met.best_accuracy(validation_labels, predictions)
    accuracy_array.append(acc)
    th_array.append(th)
    print(f"Accuracy: {acc}")

    f1_score = met.f1_score(th, validation_labels, predictions)
    f1_array.append(f1_score)
    print(f"F1 Score: {f1_score}")

    eq_acc_agn, eq_acc_psr, eq_th = met.best_eq_accuracy(validation_labels, predictions)
    eq_th_array.append(eq_th)
    eq_acc_agn_array.append(eq_acc_agn)
    eq_acc_psr_array.append(eq_acc_psr)
    print(f"EqAcc AGN: {eq_acc_agn}")
    print(f"EqAcc PSR: {eq_acc_psr}")

    acc_agn, acc_psr = met.class_accuracy(th, validation_labels, predictions)
    acc_agn_array.append(acc_agn)
    acc_psr_array.append(acc_psr)
    print(f"Accuracy AGN: {acc_agn} Accuracy PSR: {acc_psr}")

    th_pred = (predictions >= th).astype(int)
    cm_sing = met.sk_metrics.confusion_matrix(validation_labels, th_pred)
    print(cm_sing)
    cm_array.append(cm_sing)
    print("------------------------------------------------------------------------")

    modelpath = custom_paths.dir_models_path / f"TripleFolf{FOLD_NO}.keras"
    reset_model.save(modelpath)

    FOLD_NO = FOLD_NO + 1
# end for
cm_array = np.array(cm_array)

logger.debug("Training End.")
print(f"Best Model Was: {np.argmax(f1_array)}(Based on F1Score)")
print(f"Tresholds: {th_array}")
print(f"Tresholds used: {th_array[np.argmax(f1_array)]}")
print("------------------------------------------------------------------------")
print(f"Dense Layer: {best_model.layers[3].units}")
print(f"Dropout Rate: {best_model.layers[6].rate}")
print("------------------------------------------------------------------------")
print("Prediction on Evaluation DataSet")
print(f"> Loss: {np.mean(loss_array)}(+- {np.std(loss_array)})")
print(f"> AUC: {np.mean(auc_array)} (+- {np.std(auc_array)})")
print(f"> Accuracy: {np.mean(accuracy_array)} (+- {np.std(accuracy_array)})")
print(f"> F1: {np.mean(f1_array)} (+- {np.std(f1_array)})")
print(f"> EqAcc AGN: {np.mean(eq_acc_agn_array)} (+- {np.std(eq_acc_agn_array)})")
print(f"> EqAcc PSR: {np.mean(eq_acc_psr_array)} (+- {np.std(eq_acc_psr_array)})")
print(f"> Acc AGN: {np.mean(acc_agn_array)} (+- {np.std(acc_agn_array)})")
print(f"> Acc PSR: {np.mean(acc_psr_array)} (+- {np.std(acc_psr_array)})")
print("Confution Matrix")
print(
    f"{np.mean(cm_array[:,0,0])}+-{np.std(cm_array[:,0,0])}\t{np.mean(cm_array[:,0,1])}+-{np.std(cm_array[:,0,1])}"
)
print(
    f"{np.mean(cm_array[:,1,0])}+-{np.std(cm_array[:,1,0])}\t{np.mean(cm_array[:,1,1])}+-{np.std(cm_array[:,1,1])}"
)
print("------------------------------------------------------------------------")

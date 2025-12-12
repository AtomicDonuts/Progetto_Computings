import sys

import numpy as np
import pandas as pd
import keras
import keras_tuner as kt
from scipy import stats
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append("../imports/")
import nn_models as ann
import custom_variables as custom_paths
import metrics as met

df = pd.read_csv(custom_paths.csv_path)
df = df[(df["CLASS_GENERIC"] == "AGN") | (df["CLASS_GENERIC"] == "Pulsar")]
print(f"Sample Size: {len(df)}")


df["PowerLaw"] = np.where(df["SpectrumType"] == "PowerLaw",1,0,)
df["LogParabola"] = np.where(df["SpectrumType"] == "LogParabola",1,0,)
df["PLSuperExpCutoff"] = np.where(df["SpectrumType"] == "PLSuperExpCutoff",1,0,)


col_input1 = ["GLAT", "PowerLaw","LogParabola","PLSuperExpCutoff"]

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
print(f"Additionl Size: {input_additional.shape}")
print(f"Flux_Band Size: {input_flux_band.shape}")
print(f"Flux_History Size: {input_flux_hist.shape}")


is_agn = df["CLASS_GENERIC"].to_numpy() == "AGN"

labels = np.zeros((len(df)), dtype=int)
labels[~is_agn] = 1

labels_double = np.zeros((len(df), 2), dtype=int)
labels_double[is_agn, 0] = 1
labels_double[~is_agn, 1] = 1


class_weight = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(labels), y=labels
)
class_weight = {index: value for index, value in enumerate(class_weight)}

skf = StratifiedKFold(n_splits=2, shuffle=True)
train, test = next(skf.split(np.zeros(len(labels)), labels))

fb = input_flux_band[train]
hb = input_flux_hist[train]
ia = input_additional[train]
lab = labels[train]
vfb =  input_flux_band[test]
vhb =  input_flux_hist[test]
via = input_additional[test]
vlab = labels[test]


tuner = kt.Hyperband(
    ann.hp_final_model,
    objective="val_loss",
    max_epochs=100,
    factor=3,
    overwrite=False,
    directory="Finale",
    project_name="TriploInput",
)
stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
tuner.search(
    x=[fb, hb, ia],
    y=lab,
    epochs=50,
    validation_split=0.5,
    class_weight=class_weight,
    callbacks=[stop_early],
)


best_model = tuner.get_best_models(num_models=1)[0]
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
history = best_model.fit(
    x=[fb, hb, ia],
    y=lab,
    epochs=300,
    validation_split=0.5,
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr],
)
print("------------------------------------------------")
print("Dataset Separato")
predictions = best_model.predict([vfb, vhb, via])
acc, th = met.best_accuracy(vlab, predictions)
print(f"Accuracy: {acc}")
print(f"F1 Score: {met.f1_score(th,vlab, predictions)}")
acc_agn, acc_psr = met.class_accuracy(th, vlab, predictions)
print(f"Accuracy AGN: {acc_agn} Accuracy PSR: {acc_psr}")
th_pred = (predictions >= th).astype(int)
print(met.sk_metrics.confusion_matrix(vlab, th_pred))
print("------------------------------------------------")
print("Tutto il Dataset")
print("Dataset Separato")
predictions = best_model.predict([input_flux_band, input_flux_hist, input_additional])
acc, th = met.best_accuracy(labels, predictions)
print(f"Accuracy: {acc}")
print(f"F1 Score: {met.f1_score(th,labels, predictions)}")
acc_agn, acc_psr = met.class_accuracy(th, labels, predictions)
print(f"Accuracy AGN: {acc_agn} Accuracy PSR: {acc_psr}")
th_pred = (predictions >= th).astype(int)
print(met.sk_metrics.confusion_matrix(labels, th_pred))
print("------------------------------------------------")

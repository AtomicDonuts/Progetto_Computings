# %%
import sys
print("Local Machine Detected")
sys.path.append("../imports/")
import nn_models as ann
import custom_variables as custom_paths
import metrics as met
# %%
import numpy as np
import pandas as pd
import keras
import keras_tuner as kt
from scipy import stats
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# %%
df = pd.read_csv(custom_paths.csv_path)
df = df[(df["CLASS_GENERIC"] == "AGN") | (df["CLASS_GENERIC"] == "Pulsar")]
print(f"Dataset Dimentions: {len(df)}")

df["PowerLaw"] = np.where(df["SpectrumType"] == "PowerLaw",1,0,)
df["LogParabola"] = np.where(df["SpectrumType"] == "LogParabola",1,0,)
df["PLSuperExpCutoff"] = np.where(df["SpectrumType"] == "PLSuperExpCutoff",1,0,)

norm_cols = ["GLAT", "PowerLaw","LogParabola","PLSuperExpCutoff" ,"Variability_Index"]
input_datas = df[norm_cols].to_numpy()

is_agn = df["CLASS_GENERIC"].to_numpy() == "AGN"
is_psr = df["CLASS_GENERIC"].to_numpy() == "Pulsar"
labels = np.zeros((len(df)), dtype=int)
labels[is_agn] = 0
labels[is_psr] = 1

class_weight = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(labels), y=labels
)
class_weight = {index: value for index, value in enumerate(class_weight)}

skf = StratifiedKFold(n_splits=2, shuffle=True)
train, test = next(skf.split(np.zeros(len(labels)), labels))

in_train = input_datas[train]
lab_train = labels[train]
in_test = input_datas[test]
lab_test = labels[test]

# %%
tuner = kt.Hyperband(
    ann.hp_model_lr,
    objective="val_loss",
    max_epochs=100,
    factor=3,
    overwrite=False,
    directory="Finale",
    project_name=f"Confronto",
)
stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
tuner.search(
    in_train,
    lab_train,
    epochs=50,
    validation_split=0.5,
    class_weight=class_weight,
    callbacks=[stop_early],
)

# %%
# tuner.results_summary()

# %%
best_model = tuner.get_best_models(num_models=1)[0]
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
history = best_model.fit(
    in_train,
    lab_train,
    epochs=300,
    validation_split=0.5,
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr],
)

# %%
print("------------------------------------------------")
print("Dataset Separato")
predictions = best_model.predict(in_test)
acc, th = met.best_accuracy(lab_test, predictions)
print(f"Accuracy: {acc}")
print(f"F1 Score: {met.f1_score(th,lab_test, predictions)}")
acc_agn, acc_psr = met.class_accuracy(th, lab_test, predictions)
print(f"Accuracy AGN: {acc_agn} Accuracy PSR: {acc_psr}")
th_pred = (predictions >= th).astype(int)
print(met.sk_metrics.confusion_matrix(lab_test, th_pred))
print("------------------------------------------------")
print("Tutto il Dataset")
print("Dataset Separato")
predictions = best_model.predict(input_datas)
acc, th = met.best_accuracy(labels, predictions)
print(f"Accuracy: {acc}")
print(f"F1 Score: {met.f1_score(th,labels, predictions)}")
acc_agn, acc_psr = met.class_accuracy(th, labels, predictions)
print(f"Accuracy AGN: {acc_agn} Accuracy PSR: {acc_psr}")
th_pred = (predictions >= th).astype(int)
print(met.sk_metrics.confusion_matrix(labels, th_pred))
print("------------------------------------------------")

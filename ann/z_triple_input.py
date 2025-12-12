import sys

# pylint: disable=import-error, wrong-import-position
import numpy as np
import pandas as pd
import keras
from keras.models import load_model, clone_model
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
# pylint: enable=import-error, wrong-import-position

df = pd.read_csv(custom_paths.csv_path)
df = df[(df["CLASS_GENERIC"] == "AGN") | (df["CLASS_GENERIC"] == "Pulsar")]
print(f"Sample Size: {len(df)}")

df["PowerLaw"] = np.where(df["SpectrumType"] == "PowerLaw",1,0,)
df["LogParabola"] = np.where(df["SpectrumType"] == "LogParabola",1,0,)
df["PLSuperExpCutoff"] = np.where(df["SpectrumType"] == "PLSuperExpCutoff",1,0,)

col_input1 = ["GLAT","Variability_Index" ,"PowerLaw","LogParabola","PLSuperExpCutoff"]

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

class_weight = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(labels), y=labels
)
class_weight = {index: value for index, value in enumerate(class_weight)}

splitdata = StratifiedKFold(n_splits=4, shuffle=True)
train, test = next(splitdata.split(np.zeros(len(labels)), labels))

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
    directory="Progetto",
    project_name="TripleInput",
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
best_lr = best_model.optimizer.learning_rate

loss_k_array = []
auc_k_array = []
accuracy_k_array = []
acc_agn_k_array = []
acc_psr_k_array = []
f1_k_array = []
th_k_array = []
cm_k_array = []

loss_all_array = []
auc_all_array = []
accuracy_all_array = []
acc_agn_all_array = []
acc_psr_all_array = []
f1_all_array = []
th_all_array = []
cm_all_array = []

fold_no = 0
skf = StratifiedKFold(n_splits=10, shuffle=True)
for ktrain, ktest in skf.split(np.zeros(len(lab)), lab):
    k_hb   = hb[ktrain]
    k_fb   = fb[ktrain]
    k_ia   = ia[ktrain]
    k_lab  = lab[ktrain]
    k_vfb  =  hb[ktest]
    k_vhb  =  fb[ktest]
    k_via  =  ia[ktest]
    k_vlab = lab[ktest]

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

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

    history = reset_model.fit(
        x=[k_fb, k_hb, k_ia],
        y=k_lab,
        epochs=300,
        validation_data=[[k_vfb, k_vhb, k_via], k_vlab],
        callbacks=[early_stopping, reduce_lr],
    )

    print(f"Fold No.{fold_no}")
    print("------------------------------------------------------------------------")
    print("Prediction on Fold")
    scores = reset_model.evaluate([k_vfb, k_vhb, k_via], k_vlab, verbose=0)
    predictions = reset_model.predict([k_vfb, k_vhb, k_via])

    loss_k_array.append(scores[0])
    auc_k_array.append(scores[1])
    print(f"Loss: {scores[0]}")
    print(f"AUC {scores[2]}")

    acc, th = met.best_accuracy(k_vlab, predictions)
    accuracy_k_array.append(acc)
    th_k_array.append(th)
    print(f"Accuracy: {acc}")

    f1_score = met.f1_score(th,k_vlab, predictions)
    f1_k_array.append(f1_score)
    print(f"F1 Score: {f1_score}")

    acc_agn, acc_psr = met.class_accuracy(th, k_vlab, predictions)
    acc_agn_k_array.append(acc_agn)
    acc_psr_k_array.append(acc_psr)
    print(f"Accuracy AGN: {acc_agn} Accuracy PSR: {acc_psr}")

    th_pred = (predictions >= th).astype(int)
    cm_sing = met.sk_metrics.confusion_matrix(k_vlab, th_pred)
    print(cm_sing)
    cm_k_array.append(cm_sing)

    print("------------------------------------------------------------------------")

    print("Prediction on Full Dataset")
    scores = reset_model.evaluate([input_flux_band, input_flux_hist, input_additional], labels, verbose=0)
    predictions = reset_model.predict([input_flux_band, input_flux_hist, input_additional])

    loss_all_array.append(scores[0])
    auc_all_array.append(scores[1])
    print(f"Loss: {scores[0]}")
    print(f"AUC {scores[2]}")

    acc, th = met.best_accuracy(labels, predictions)
    accuracy_all_array.append(acc)
    th_all_array.append(th)
    print(f"Accuracy: {acc}")

    f1_score = met.f1_score(th,labels, predictions)
    f1_all_array.append(f1_score)
    print(f"F1 Score: {f1_score}")

    acc_agn, acc_psr = met.class_accuracy(th, labels, predictions)
    acc_agn_all_array.append(acc_agn)
    acc_psr_all_array.append(acc_psr)
    print(f"Accuracy AGN: {acc_agn} Accuracy PSR: {acc_psr}")

    th_pred = (predictions >= th).astype(int)
    cm_sing = met.sk_metrics.confusion_matrix(labels, th_pred)
    print(cm_sing)
    cm_all_array.append(cm_sing)
    print("------------------------------------------------------------------------")

    modelpath = custom_paths.dir_models_path / f"TripleFolf{fold_no}.keras"
    reset_model.save(modelpath)

    fold_no = fold_no + 1
# end for
print("------------------------------------------------------------------------\n")
print("Average scores for all folds:")
print("Prediction on Fold")
print(f"> Loss: {np.mean(loss_k_array)}")
print(f"> AUC: {np.mean(auc_k_array)} (+- {np.std(auc_k_array)})")
print(f"> Accuracy: {np.mean(accuracy_k_array)} (+- {np.std(accuracy_k_array)})")
print(f"> F1: {np.mean(f1_k_array)} (+- {np.std(f1_k_array)})")
print(f"> Acc AGN: {np.mean(acc_agn_k_array)} (+- {np.std(acc_agn_k_array)})")
print(f"> Acc PSR: {np.mean(acc_psr_k_array)} (+- {np.std(acc_psr_k_array)})")
print("--------------------------------------------------------")
print("Prediction on All DataSet")
print(f"> Loss: {np.mean(loss_all_array)}")
print(f"> AUC: {np.mean(auc_all_array)} (+- {np.std(auc_all_array)})")
print(f"> Accuracy: {np.mean(accuracy_all_array)} (+- {np.std(accuracy_all_array)})")
print(f"> F1: {np.mean(f1_all_array)} (+- {np.std(f1_all_array)})")
print(f"> Acc AGN: {np.mean(acc_agn_all_array)} (+- {np.std(acc_agn_all_array)})")
print(f"> Acc PSR: {np.mean(acc_psr_all_array)} (+- {np.std(acc_psr_all_array)})")
print("------------------------------------------------------------------------")

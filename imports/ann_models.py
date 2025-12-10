"""
doc
"""

import keras
from keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential


def paper_model(flux_band_shape, flux_hist_shape):
    """
    doc string
    """
    inputs = Input(shape=flux_band_shape)
    hidden = Flatten()(inputs)
    hidden = Dense(16, activation="relu")(hidden)
    inputs2 = Input(shape=flux_hist_shape)
    hidden2 = Flatten()(inputs2)
    hidden2 = Dense(16, activation="relu")(hidden2)
    concat = Concatenate()([hidden2, hidden])
    final_hidden = Dense(4, activation="relu")(concat)
    outputs = Dense(2, activation="softmax")(final_hidden)
    model = Model(inputs=[inputs, inputs2], outputs=outputs)
    return model


def simple_model(input_data_shape):
    """
    doc
    """
    model = Sequential()
    model.add(Input(shape=input_data_shape))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def hp_model(hp):
    """
    doc
    """
    model = Sequential()

    nodes = hp.Int("nodes", min_value=4, max_value=16, step=4)
    drops = hp.Float("dropout", min_value=0.2, max_value=0.6, step=0.1)

    model.add(Dense(units=nodes, activation="relu"))
    model.add(Dropout(rate=drops))

    model.add(Dense(units=nodes * 2, activation="relu"))
    model.add(Dropout(rate=drops))

    model.add(Dense(units=nodes, activation="relu"))
    model.add(Dropout(rate=drops))

    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            #            "f1_score",
            "auc",
        ],
    )
    return model


def hp_model_lr(hp):
    """
    doc
    """
    model = Sequential()

    nodes = hp.Choice("nodes", [4, 8, 16, 32])
    drops = hp.Float("dropout", min_value=0.2, max_value=0.6, step=0.1)
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    model.add(Dense(units=nodes, activation="relu"))
    model.add(Dropout(rate=drops))

    model.add(Dense(units=nodes * 2, activation="relu"))
    model.add(Dropout(rate=drops))

    model.add(Dense(units=nodes, activation="relu"))
    model.add(Dropout(rate=drops))

    model.add(Dense(units=int(nodes / 2), activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            #"f1_score",
            "auc",
        ],
    )
    model.optimizer.learning_rate = learning_rate
    return model

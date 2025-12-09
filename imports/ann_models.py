"""
doc
"""

import numpy as np
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

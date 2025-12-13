"""
This module defines the Keras architectures for the Neural Networks used in the project.
It includes simple models, multi-input models, and functions for hyperparameter tuning.
"""

from keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential


def paper_model(flux_band_shape, flux_hist_shape):
    """
    Builds a multi-input neural network model based on the reference paper architecture.
    It takes flux bands and flux history as separate inputs.

    :param flux_band_shape: Shape of the flux band input features.
    :type flux_band_shape: tuple
    :param flux_hist_shape: Shape of the flux history input features.
    :type flux_hist_shape: tuple
    :return: A compiled Keras Model.
    :rtype: keras.models.Model
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
    Builds a simple sequential Feed-Forward Neural Network.

    :param input_data_shape: Shape of the input features.
    :type input_data_shape: tuple
    :return: A uncompiled Keras Sequential Model.
    :rtype: keras.models.Sequential
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

def hp_model_lr(hp):
    """
    Builds a tunable sequential model that also tunes the learning rate.

    :param hp: Hyperparameters object from Keras Tuner.
    :type hp: keras_tuner.HyperParameters
    :return: A compiled Keras Model with tunable learning rate.
    :rtype: keras.models.Sequential
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
            "auc",
        ],
    )
    model.optimizer.learning_rate = learning_rate
    return model


def final_model(flux_band_shape, flux_hist_shape, input_data_shape):
    """
    Constructs the final three-input Neural Network architecture.
    Inputs are: Flux Band, Flux History, and Additional Features (e.g., Coordinates, Index).

    :param flux_band_shape: Shape tuple for Flux Band input.
    :type flux_band_shape: tuple
    :param flux_hist_shape: Shape tuple for Flux History input.
    :type flux_hist_shape: tuple
    :param input_data_shape: Shape tuple for Additional Features input.
    :type input_data_shape: tuple
    :return: A compiled Keras Model.
    :rtype: keras.models.Model
    """
    inputs = Input(shape=flux_band_shape)
    hidden = Dense(16, activation="relu")(inputs)
    hidden = Dropout(0.2)(hidden)
    hidden = Dense(4, activation="relu")(hidden)

    inputs2 = Input(shape=flux_hist_shape)
    hidden2 = Dense(16, activation="relu")(inputs2)
    hidden2 = Dropout(0.2)(hidden2)
    hidden2 = Dense(4, activation="relu")(hidden2)

    inputs3 = Input(shape=input_data_shape)
    hidden3 = Dense(16, activation="relu")(inputs3)
    hidden3 = Dropout(0.2)(hidden3)
    hidden3 = Dense(4, activation="relu")(hidden3)

    concat = Concatenate()([hidden, hidden2, hidden3])
    final_hidden = Dense(32, activation="relu")(concat)
    final_hidden = Dense(4, activation="relu")(final_hidden)
    outputs = Dense(1, activation="sigmoid")(final_hidden)

    model = Model(
        inputs=[
            inputs,
            inputs2,
            inputs3
        ],
        outputs=outputs,
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy",
                 #"f1_score",
                 "auc"],
    )
    model.optimizer.learning_rate = 0.01
    return model


def hp_final_model(hp):
    """
    Constructs a tunable version of the final three-input architecture for Keras Tuner.
    Allows tuning of layer sizes, dropout, and learning rate.

    :param hp: Hyperparameters object from Keras Tuner.
    :type hp: keras_tuner.HyperParameters
    :return: A compiled Keras Model ready for tuning.
    :rtype: keras.models.Model
    """
    nodes = hp.Choice("nodes", [4, 8, 16, 32])
    drops = hp.Float("dropout", min_value=0.2, max_value=0.6, step=0.1)
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    inputs = Input(shape=(16,))
    hidden = Dense(units = nodes * 2 , activation="relu")(inputs)
    hidden = Dropout(rate = drops)(hidden)
    hidden = Dense(units = nodes, activation="relu")(hidden)

    inputs2 = Input(shape=(28,))
    hidden2 = Dense(units=nodes * 2, activation="relu")(inputs2)
    hidden2 = Dropout(rate=drops)(hidden2)
    hidden2 = Dense(units=nodes, activation="relu")(hidden2)

    inputs3 = Input(shape=(5,))
    hidden3 = Dense(units=nodes * 2, activation="relu")(inputs3)
    hidden3 = Dropout(rate=drops)(hidden3)
    hidden3 = Dense(units=nodes, activation="relu")(hidden3)

    concat = Concatenate()([hidden, hidden2, hidden3])
    final_hidden = Dense(units=nodes * 2, activation="relu")(concat)
    final_hidden = Dropout(rate=drops)(final_hidden)
    final_hidden = Dense(units=nodes, activation="relu")(final_hidden)
    final_hidden = Dense(units=4, activation="relu")(final_hidden)
    outputs = Dense(1, activation="sigmoid")(final_hidden)

    model = Model(
        inputs=[inputs, inputs2, inputs3],
        outputs=outputs,
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            "auc",
        ],
    )
    model.optimizer.learning_rate = learning_rate
    return model

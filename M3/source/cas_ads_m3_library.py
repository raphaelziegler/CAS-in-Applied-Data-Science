#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File name: cas_ads_m3_library.py

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_addons as tfa


def versions():
    print(f"pandas {pd.__version__}")
    print(f"numpy {np.__version__}")
    print(f"matplotlib {matplotlib.__version__}")
    print(f"sklearn {sklearn.__version__}")
    print(f"tensorflow {tf.__version__}")


def create_test_train_set(x: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Generate the train and test sets

    :param x: Dataframe with the x values (input)
    :param y: Dataframe with the y values (output)
    :param test_size: Split size 20% of the total size for testing
    :return: train, test values as Dataframe
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


class LinearModel:
    """
    Provide all the necessary moduls to work with linear regression.
    """

    def linear_model(self, x: pd.DataFrame, y: pd.DataFrame) -> LinearRegression:
        """
        Create the linear model object and do the fitting

        :param x: Dataframe with the x values (x_train)
        :param y: Dataframe with the y values (y_train)
        :return: linear regression model object
        """
        reg = linear_model.LinearRegression()  # create the model
        reg.fit(x, y)  # do the fitting
        return reg

    def mse(self, x: pd.DataFrame, y: pd.DataFrame, reg: LinearRegression) -> float:
        """
        Calculate the mean squared error

        :param x: Dataframe with the x values
        :param y: Dataframe with the y values
        :param reg: linear regression model object
        :return: Mean squared error
        """
        return np.std(y - reg.predict(x))[0]

    def r2(self, x: pd.DataFrame, y: pd.DataFrame, reg: LinearRegression) -> float:
        """
        Calculate the R-squared value

        :param x: Dataframe with the x values
        :param y: Dataframe with the y values
        :param reg: linear regression model object
        :return: R-squared (R2)
        """
        return reg.score(x, y)


class NeuralNetwork:
    """
    Provide all the necessary moduls to work with the neural network
    """

    def creat_network(self, input_shape: int) -> tf.keras.models.Sequential:
        """
        Create the neural network model and define the training
        (loss function, metrics)

        :param input_shape: Number of input variables
        :return: neural network object
        """

        # create the model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(input_shape,)),  # input shape
            # 1. layer: number of neurons, activations function relu
            tf.keras.layers.Dense(128, activation='relu'),
            # 2. layer
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            # last layer has 1 neuron and no activation function as we're
            # looking for a number value
            tf.keras.layers.Dense(1, activation=None)])

        # Train the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss='mse',  # what loss function to use
                      metrics=[tfa.metrics.r_square.RSquare()])  # what metrics we want tu use
        # model.summary()
        return model

    def fit_model(self, model, x_train: pd.DataFrame, y_train: pd.DataFrame,
                  x_test: pd.DataFrame, y_test: pd.DataFrame, epoch: int = 256):
        """
        Fit the neural network and output the training data for plotting

        :param model: neural network model object
        :param x_train: training data
        :param y_train: training data
        :param x_test: test data
        :param y_test: test data
        :param epoch: number of epochs
        :return: historic model training data
        """

        # path where to save the epoch data
        save_path = 'save/col_{epoch}.ckpt'
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True)

        # fit model and save training data
        hist = model.fit(x=x_train, y=y_train,  # training data
                         epochs=epoch, batch_size=32,
                         validation_data=(x_test, y_test),  # validation data
                         callbacks=[save_callback])  # save values for each epoch

        return hist

    def show_learning_plots(self, hist):
        """
        Plot the training history

        :param hist: historic training data
        """
        # we plot 2 plots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # plot number 0
        axs[0].plot(hist.epoch, hist.history['loss'])
        axs[0].plot(hist.epoch, hist.history['val_loss'])
        axs[0].legend(('training loss', 'validation loss'), loc='upper right')
        axs[0].set_xlabel("epochs")

        # plot number 1
        axs[1].plot(hist.epoch, hist.history['r_square'])
        axs[1].plot(hist.epoch, hist.history['val_r_square'])
        axs[1].legend(('training R2', 'validation R2'))

        # add boundary line
        axs[1].hlines(y=0.7, xmin=hist.epoch[0], xmax=hist.epoch[-1],  color="black", linestyles="dotted")
        axs[1].fill_between(x=hist.epoch, y1=0, y2=0.7, color="papayawhip")

        # set y scale to 0 to 1
        axs[1].set_ylim([0, 1])

        # set label for the x axis
        axs[1].set_xlabel("epochs")
        plt.show()

    def evaluate(self, x: pd.DataFrame, y: pd.DataFrame, model):
        """
        Display model evaluation

        :param x: test data
        :param y: test data
        :param model: neural network model object
        """
        model.evaluate(x, y, verbose=2)

    def predict(self, model, x: np.array):
        """
        Predict the outputvalue depending on the input

        :param model: neural network model
        :param x: input data as np.array
        :return: predicted value
        """
        return model.predict(x)


if __name__ == '__main__':
    versions()

    # read the source file
    raw_df = pd.read_excel("colums_list.xlsx", sheet_name="Eingabe", skiprows=17)
    # set header names
    raw_df.columns = ["Ort", "Projektname", "Datum", "Bezeichnung",
                      "Stahlbetonstütze", "Stahlbetonverbundstütze", "rund",
                      "eckig",
                      "oval", "ø", "a", "b", "l", "Nd", "Md", "Stück",
                      "Hersteller", "Einkauf_LoMa", "Einkauf_Baumeister",
                      "LoMa", "Rabatt inkl. Skonto LoMa",
                      "Transport", "Stückpreis", "Baumeister",
                      "Rabatt inkl. Skonto Baumeister",
                      "Baumeiser inkl. Rabatt & Teuerung",
                      "A", "V", "N/mm2", "CHF/m3", "Teuerung", "Total Preis",
                      "Bemerkungen"]

    # filter out error data
    raw_df["Nd"] = np.where(raw_df['Nd'] > 15000, 0, raw_df["Nd"])
    raw_df["l"] = np.where(raw_df['l'] < 2.0, 0, raw_df["l"])
    df = raw_df[raw_df['Nd'] > 0]
    df = df[raw_df['l'] > 0]
    df[["Nd", "A", "V", "l", "Stückpreis"]].describe()

    x = df[["Nd", "A", "V", "l"]]
    y = df[["Stückpreis"]]

    x_train, x_test, y_train, y_test = create_test_train_set(x, y)

    LM = LinearModel()
    LMmodel = LM.linear_model(x_train, y_train)
    print("MSE:")
    print(LM.mse(x_train, y_train, LMmodel))
    print(LM.mse(x_test, y_test, LMmodel))
    print("R2:")
    print(LM.r2(x_train, y_train, LMmodel))
    print(LM.r2(x_test, y_test, LMmodel))

    NN = NeuralNetwork()
    NNmodel = NN.creat_network(4)

    hist = NN.fit_model(NNmodel, x_train, y_train, x_test, y_test)

    # NN.show_learning_plots(hist)

    NN.evaluate(x_test, y_test, NNmodel)

    xin = np.array([[2500, .25 * .25, .25 * .25 * 3, 3]])
    r = NN.predict(NNmodel, xin)
    print(r)

import tensorflow as tf
import numpy as np
import math
import os

#compression
import pickle
import bz2

import matplotlib.pyplot as plt


class Model:
    """Create a fully connected Neural Network

    Validation set validates our model performance during training,
    reducing problems such as overfitting

    Not using a CNN because input is condensed into numpy array

    Parameters
    ----------
    train_x : (Type)
        Training ECG data
    train_y : (Type)
        Training ECG labels
    test_x : (Type)
        Testing ECG data
    test_y : (Type)
        Testing ECG labels

    Returns
    -------
    (Type)
        A NN model

    """

    def init(self, train_X, train_y, test_X, test_y):
        self.compiled = self.compile()

        #if not an np array, convert to np.asarray(train_X)
        print(type(trainX))

        #taking 25%
        self.validation_X = np.asarray(test_X[( (len(test_X)//2) //2):])
        self.test_X = np.asarray(test_X[:( (len(test_X)//2) //2)])

        #taking 25%
        self.validation_y = np.asarray(test_y[( (len(test_y)//2) //2):])
        self.test_y = np.array(test_y[:( (len(test_y)//2) //2)])


    def compile(self):
        """Creates a sequental where every node has exactly one input and output tensor

        Returns
        -------
        (Type)
            Compiled model
        """
        #Figure out which layers we are looking for in our model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten()) #flattens input
        model.add(tf.keras.layers.Dense(1, activation = tf.nn.relu)) #relu activation


        model.add(tf.keras.layers.Dropout(0.5)) #prevent overfitting

        optimize = tf.keras.optimizers.Adam(learning_rate = 0.001)

        model.compile(
        optimizer = optimize,
        loss = "mean_squared_error", #Seek to change this?
        metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
        )

        return model


class Train:

    """Trains the NN

    Parameters
    ----------
    model : (Type)
        A NN model

    Returns
    -------
    compiled_model : (Type)
        A compiled NN model

    """

    def __init__(self,model):
        self.model = model
        self.compiled_model = model.compiled_model
        self.trained_model = self.train_model(self.compiled_model, self.model)

    def train_model(self, compiled_model, model):
        """Trains NN model

            Returns compiled model
        """
        #EarlyStopping??
        tf.keras.calbacks.EarlyStopping(
        monitor = "val_loss",
        mode = "min",
        patience = 20,
        restore_best_weights = True
        )

        compiled_model.fit(
        model.train_X,
        model.train_y,
        epochs = 100,
        validation_data = (model.validation_x, model.validation_y),
        callbacks = [earlystopping])


        return compiled_model

class Evaluate:
    """Evaluates model performance

        Parameters:
        -----------
        compiled_model :
            trained model
        trained_model : (Type)
            trained model

        Returns:
        --------
        val_loss : (Type)
            validation loss???
        val_rmse :

        val_mae :

    """

    def __init__(self,compiled_model, trained_model):
        self.compiled_model = compiled_model
        self.trained_model = trained_model

        self.loss, self.rmse, self.mae = trained_model.evaluate(compiled_model.test_X,compiled_model.test_y)

def get_runtime(func):

    def time_func():
        print()
        func()
    return time_func


def splitExamples(data,labels):
    """Splits data and labels into separate training / testing arrays

        Parameters:
        -----------
        data : np array
            Data __
        labels : np array
            __

        Returns:
        --------
        train_X : np array
            training examples
        train_y: np array
            training examples
        test_X: np array
            # TEMP:
        test_y: np array
            sd
    """

    #going to split 80/20 for now

    print(len(data))

    trainx = 0
    trainy = 0

    testx = 0
    testy = 0


    return trainx, trainy, testx, testy



if __name__ == "__main__":

    #loading data and labels
    in_file = bz2.BZ2File("/Users/lukelorenz/Desktop/ECGNN/sim_ecg_data.bz2",'rb')
    data = pickle.load(in_file)
    in_file.close()

    in_file = bz2.BZ2File("/Users/lukelorenz/Desktop/ECGNN/sim_ecg_labels.bz2",'rb')
    labels = pickle.load(in_file)
    in_file.close()


    x_train, y_train, x_test, y_test = splitExamples(data,labels)
    print(len(x_train))

    #compiled_model = Model(x_train, y_train, x_test, y_test)

    #trained_model = Train(compiled_model)

    #trained_model = trained_model.trained_model

    #evaluate = Evaluate(compiled_model,trained_model)
    #print(evaluate)

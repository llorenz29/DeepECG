import tensorflow as tf
import numpy as np
import math
import os

#compression
import pickle
import bz2

#timing
import time

import matplotlib.pyplot as plt


class Model:
    """Create a fully connected Neural Network

    Validation set validates our model performance during training,
    reducing problems such as overfitting

    Not using a CNN because input is condensed into numpy array

    May get the warning message:
        'This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use
        the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2
        To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags'

        Which is not an error, just saying can take advantange of other CPU optimizations

    Parameters
    ----------
    train_x : numpy.ndarray
        Training ECG data
    train_y : numpy.ndarray
        Training ECG labels
    test_x : numpy.ndarray
        Testing ECG data
    test_y : numpy.ndarray
        Testing ECG labels

    Returns
    -------
    Sequential
        A NN model

    """

    def __init__(self, train_X, train_y, test_X, test_y):
        #self.compiled = our NN model
        self.compiled = self.compile()

        #taking 50% of testing for a split of 70 training 15 validation 15 testing
        self.validation_X = test_X[( (len(test_X)//2) ):]
        self.test_X = test_X[:( (len(test_X)//2) )]

        self.validation_y = test_y[( (len(test_y)//2) ):]
        self.test_y = test_y[:( (len(test_y)//2) )]



    def compile(self):
        """Creates a fully connected sequental NN model where every node has exactly one input and output tensor

        Returns
        -------
        Sequential
            Compiled model
        """


        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape =(12,2500))) #flattens input of shape (example, 12,2500) for 12 lead, 2500 sample duration
        model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu)) #regular dense NN layer
        model.add(tf.keras.layers.Dropout(0.5)) #prevent overfitting
        model.add(tf.keras.layers.Dense(1))

        print(model.output_shape)

        #using Adam optimizer, mse, and metrics
        model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = "mean_squared_error", #Seek to change this?
        metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
        )

        return model


class Train:

    """Trains the NN

    Parameters
    ----------
    model : Sequential
        A NN model

    Returns
    -------
    compiled_model : Sequential
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
        compiled_model : keras Model class
            trained model
        trained_model : keras Model class
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
    """Decorator to get the various runtimes of different functions

        Runs specified input function and times execution time
    """
    def time_func(*args,**kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"The function {func.__name__} took {end-start} seconds to run")
        return res
    return time_func



@get_runtime
def splitExamples(data,labels,split_factor):
    """Splits data and labels into separate training / testing arrays

        Parameters:
        -----------
        data : np array
            ECG waveform data
        labels : np array
            labels associated with an example
        split_factor : float range (0-1)
            What proportion of examples will be training (0.8 = 80%)

        Returns:
        --------
        train_X : numpy.ndarray
            training examples
        train_y: numpy.ndarray
            training labels
        test_X: numpy.ndarray
            testing examples
        test_y: numpy.ndarray
            testing labels
    """

    #going to split 70/30 for now (70 train, 15 val, 15 test)

    #splitting training data
    trainx = data[:int(len(data)*split_factor)]
    trainy = labels[:int(len(labels)*split_factor)]

    #splitting testing data
    testx = data[int(len(data)*split_factor):]
    testy = labels[int(len(labels)*split_factor):]

    return trainx, trainy, testx, testy



if __name__ == "__main__":

    #loading data and labels
    in_file = bz2.BZ2File("/Users/lukelorenz/Desktop/ECGNN/sim_ecg_data.bz2",'rb')
    data = pickle.load(in_file)
    in_file.close()

    in_file = bz2.BZ2File("/Users/lukelorenz/Desktop/ECGNN/sim_ecg_labels.bz2",'rb')
    labels = pickle.load(in_file)
    in_file.close()

    #print(data)
    #print(labels)


    x_train, y_train, x_test, y_test = splitExamples(data,labels, 0.7)


    compiled_model = Model(x_train, y_train, x_test, y_test)

    compiled_model.compile()

    #trained_model = Train(compiled_model)

    #trained_model = trained_model.trained_model

    #evaluate = Evaluate(compiled_model,trained_model)
    #print(evaluate)

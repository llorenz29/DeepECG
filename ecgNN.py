import tensorflow as tf
import numpy as np
import math
import os
import sklearn

#compression
import pickle
import bz2

#timing
import time

#plotting
import matplotlib.pyplot as plt


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



class Model:
    """Create a Convolutional Neural Network with outputs into dense layers

    Validation set validates our model performance during training,
    reducing problems such as overfitting


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
        A CNN model

    """

    def __init__(self, train_X, train_y, test_X, test_y):

        #compiled network
        self.compiled = self.compile()

        self.train_X = train_X
        self.train_y = train_y

        self.test_X = test_X
        self.test_y = test_y

        #taking 50% of testing for a split of 70 training 15 validation 15 testing
        self.validation_X = test_X[( (len(test_X)//2) ):]
        self.test_X = test_X[:( (len(test_X)//2) )]

        self.validation_y = test_y[( (len(test_y)//2) ):]
        self.test_y = test_y[:( (len(test_y)//2) )]



    def compile(self):
        """

        """
        input_layer = tf.keras.Input(shape=(12, 2500))

        #expanding output space
        x = tf.keras.layers.Conv1D(
            filters=32, kernel_size=3, strides=2, activation="relu", padding="same"
        )(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, strides=2, activation="relu", padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv1D(
        filters=128, kernel_size=5, strides=2, activation="relu", padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv1D(
        filters=256, kernel_size=5, strides=2, activation="relu", padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv1D(
        filters=512, kernel_size=7, strides=2, activation="relu", padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv1D(
        filters=1024, kernel_size=7, strides=2, activation="relu", padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(4096, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(
        2048, activation="relu", kernel_regularizer=tf.keras.regularizers.L2()
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(
        1024, activation="relu", kernel_regularizer=tf.keras.regularizers.L2()
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(
        128, activation="relu", kernel_regularizer=tf.keras.regularizers.L2()
        )(x)
        output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        print(model.summary())


        optimizer = tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
        loss = tf.keras.losses.BinaryCrossentropy()

        #compiling model
        model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.Accuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            ],
            )

        return model


class Train:

    """Trains our fully connected Neural Network

    Parameters:
    ----------
    model : Sequential
        A NN model, compiled as specified by the Model class above

    Stores:
    -------
    self.compliled : Sequential
        compiled CNN model containing build specified in Model class
    self.trained_model :
        Trained CNN fit on training data

    """

    def __init__(self,model):
        self.model = model
        self.compiled = model.compiled
        self.trained_model = self.train_model(self.compiled, self.model)

    @get_runtime #prints runtime to train the model
    def train_model(self, compiled, model):
        """Returns a compiled (trained) model
        """

        #training the model on training data
        #starting with 20 epochs, if still improving performance will increase
        compiled.fit(
        x = model.train_X,
        y = model.train_y,
        epochs = 30,
        validation_data = (model.validation_X, model.validation_y)
        )

        return compiled

class Evaluate:
    """Evaluates model performance on our testing data

        Parameters:
        -----------
        compiled_model : keras Model class
            compiled model from Model() class
        trained_model : keras Model class
            trained model from Train() class

        Stores:
        --------
        loss :
            BinaryCrossentropy
        accuracy :
            Accuracy metric
        auc :
            Area Under Curve
        precision :
            precision preformance metric
        recall :
            recall preformance metric

    """

    def __init__(self,compiled, trained_model):
        self.compiled = compiled
        self.trained_model = trained_model

        self.loss, self.accuracy, self.auc, self.precision, self.recall  = self.trained_model.trained_model.evaluate(compiled.test_X,compiled.test_y)




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

    print(data.shape)



    x_train, y_train, x_test, y_test = splitExamples(data,labels, 0.7)


    compiled_model = Model(x_train, y_train, x_test, y_test)

    #training takes 30 mins
    trained_model = Train(compiled_model)

    # print(trained_model.trained_model.predict(x_test))
    # print(y_test)

    print("done training")

    evaluate_model = Evaluate(compiled_model, trained_model)

    #loss, accuracy, auc, precision, recall = trained_model.trained_model.evaluate(x_test)
    print(f"Loss : {evaluate_model.loss}")
    #print(f"Accuracy : {evaluate_model.accuracy}")
    print(f"Area under the Curve (ROC) : {evaluate_model.auc}")
    print(f"Precision : {evaluate_model.precision}")
    print(f"Recall : {evaluate_model.recall}")

    #trained_model = trained_model.trained_model

    #evaluate = Evaluate(compiled_model,trained_model)
    #print(evaluate)

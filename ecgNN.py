import tensorflow as tf
import numpy as np
import math
import os
from sklearn.model_selection import train_test_split


#compression
import pickle
import bz2

#timing
import time

#plotting
import matplotlib.pyplot as plt

#import keras_tuner


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
        #self.tb = TensorBoard(log_dir = "logs/{}".format(time.time()))

        self.train_X = train_X
        self.train_y = train_y


        #taking 50% of testing for a split of 70 training 15 validation 15 testing
        self.validation_X = test_X[( (len(test_X)//2) ):]
        self.test_X = test_X[:( (len(test_X)//2) )]

        self.validation_y = test_y[( (len(test_y)//2) ):]
        self.test_y = test_y[:( (len(test_y)//2) )]



    def residual_block(self,X, num_filter, kernel_size, down_sample=False):
        print(f"num_filter: {num_filter}")
        X_shortcut = X

        if down_sample == False:
            X = tf.keras.layers.Conv1D(filters=num_filter, kernel_size=kernel_size, strides=1, padding='same')(X)
        else:
            X = tf.keras.layers.Conv1D(filters=num_filter, kernel_size=kernel_size, strides=2, padding='same')(X)
            X_shortcut = tf.keras.layers.Conv1D(filters=num_filter, kernel_size=1, strides=2, padding='same')(X_shortcut)
            X_shortcut = tf.keras.layers.BatchNormalization(axis=2)(X_shortcut)

        X = tf.keras.layers.BatchNormalization(axis=2)(X)
        X = tf.keras.layers.ReLU()(X)

        X = tf.keras.layers.Conv1D(filters=num_filter, kernel_size=kernel_size, strides=1, padding='same')(X)
        X = tf.keras.layers.BatchNormalization(axis=2)(X)

        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.ReLU()(X)

        return X

    def compile(self):
        """
        Resnet model adaptation

        """

        input_layer = tf.keras.Input(shape=(2500,12))
        # Conv1
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=7, strides=2, padding='same')(input_layer)
        # Conv2_x
        x = tf.keras.layers.MaxPooling1D(pool_size =3, strides =2, padding ='same')(x)
        x = self.residual_block(x, num_filter=64, kernel_size=3, down_sample=False)
        x = self.residual_block(x, num_filter=64, kernel_size=3, down_sample=False)
        # Conv3_x
        x = self.residual_block(x, num_filter=128, kernel_size=3, down_sample=True)
        x = self.residual_block(x, num_filter=128, kernel_size=3, down_sample=False)
        # Conv4_x
        x = self.residual_block(x, num_filter=256, kernel_size=3, down_sample=True)
        x = self.residual_block(x, num_filter=256, kernel_size=3, down_sample=False)
        # Conv5_x
        x = self.residual_block(x, num_filter=512, kernel_size=3, down_sample=True)
        x = self.residual_block(x, num_filter=512, kernel_size=3, down_sample=False)
        # Classifier
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        print(model.summary())

        #compile
        model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
              tf.keras.metrics.AUC()]
        )

        #tf.keras.utils.vis_utils.plot_model(model, show_shapes=True, show_layer_names=True)

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
        epochs = 25,
        validation_data = (model.validation_X, model.validation_y)
        )

        return compiled

    def save_model(self):

        self.compiled.save("simulation_model")

        pass

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

        self.loss, self.accuracy, self.auc  = self.trained_model.trained_model.evaluate(compiled.test_X,compiled.test_y)




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

def preprocess(X):
    print(f"x shape{X.shape[1]}")
    m=4096-X.shape[1]
    y=np.pad(X,[(0,0),(0,m),(0,0)],mode='constant', constant_values=0)
    return y

if __name__ == "__main__":


    #loading data and labels
    # in_file = bz2.BZ2File("/Users/lukelorenz/Desktop/ECGNN/data/sim_ecg_data.bz2",'rb')
    # data = pickle.load(in_file)
    # in_file.close()
    #
    # in_file = bz2.BZ2File("/Users/lukelorenz/Desktop/ECGNN/data/sim_ecg_labels.bz2",'rb')
    # labels = pickle.load(in_file)
    # in_file.close()

    data = np.load('data/sim_ecg_data.bz2')
    labels = np.load('data/sim_ecg_labels.bz2')



    print(data.shape)
    reshaped = np.transpose(data, axes=[0, 2, 1]) #switching data
    print(reshaped.shape)

    print(f"example 1: {reshaped[0]}")
    print(f"example 1 label: {labels[0]}")

    idx = np.random.permutation(len(data))
    data,labels = data[idx], labels[idx]


    x_train, y_train, x_test, y_test = splitExamples(reshaped,labels,0.8)

    x_train=preprocess(x_train)
    x_test=preprocess(x_test)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    #x_train, y_train, x_test, y_test = splitExamples(reshaped,labels, 0.7)


    compiled_model = Model(x_train, y_train, x_test, y_test)

    #training takes 30 mins
    trained_model = Train(compiled_model)

    trained_model.save_model()

    # print(trained_model.trained_model.predict(x_test))
    # print(y_test)

    print("done training")

    #saved_model = tf.keras.models.load_model('simulation_model')
    #print("loaded")
    #saved_model.summary()
    print("evaluating")
    # evaluate_model = Evaluate(compiled_model, trained_model)
    #
    #
    #
    # #loss, accuracy, auc, precision, recall = trained_model.trained_model.evaluate(x_test)
    # print(f"Loss : {evaluate_model.loss}")
    # #print(f"Accuracy : {evaluate_model.accuracy}")
    # print(f"Area under the Curve (ROC) : {evaluate_model.auc}")
    # print(f"Precision : {evaluate_model.precision}")
    # print(f"Recall : {evaluate_model.recall}")

    res = trained_model.trained_model.predict(x_test)
    print(f"res: {res}")

    loss, accuracy, auc = trained_model.trained_model.evaluate(x_test,y_test)
    print(f"Loss : {loss}")
    print(f"Binary Accuracy : {accuracy}")
    print(f"Area under the Curve (ROC) : {auc}")

    #trained_model.trained_model.summary()

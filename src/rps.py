'''Trains a simple convnet on the MNIST dataset.
based on a keras example by fchollet
Find a way to improve the test accuracy to almost 99%!
FYI, the number of layers and what they do is fine.
But their parameters and other hyperparameters could use some work.
'''
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.transform import resize, rotate
import os

import numpy as np
np.random.seed(1337)  # for reproducibility

def load_images(dir_name, data_kind):
    img_lst = []
    base_path = 'rps_images/' + data_kind
    for filename in os.listdir(base_path + '/' + dir_name):
        img_lst.append(io.imread(base_path + '/' + dir_name + '/' + filename))
    return img_lst

def load_and_featurize_data():
    # train data 
    paper_train = load_images('paper', 'train')
    rock_train = load_images('rock', 'train')
    sc_train= load_images('scissors', 'train')
    X_train = np.array(paper_train + rock_train  + sc_train)

    # test data
    paper_test = load_images('paper', 'test')
    rock_test = load_images('rock', 'test')
    sc_test = load_images('scissors', 'test')
    X_test = np.array(paper_test + rock_test  + sc_test)

    # reshape input into format Conv2D layer likes
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)

    # don't change conversion or normalization
    X_train = X_train.astype('float32')  # data was uint8 [0-255]
    X_test = X_test.astype('float32')    # data was uint8 [0-255]
    X_train /= 255  # normalizing (scaling from 0 to 1)
    X_test /= 255   # normalizing (scaling from 0 to 1)


    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    y_train = np.array([3*i//X_train.shape[0] for i in range(X_train.shape[0])])
    y_test = np.array([3*i//X_test.shape[0] for i in range(X_test.shape[0])])

    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, nb_classes)  # cool
    Y_test = to_categorical(y_test, nb_classes)
    
    # Shuffle all data
    splits = [X_train, X_test, y_train, y_test]
    rng_state = np.random.get_state()
    for split in splits:
        np.random.shuffle(split)
        np.random.set_state(rng_state)


    return X_train, X_test, Y_train, Y_test


def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential()  # model is a linear stack of layers (don't change)

    # note: the convolutional layers and dense layers require an activation function
    # see https://keras.io/activations/
    # and https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     input_shape=input_shape))  # first conv. layer  KEEP
    model.add(Activation('relu'))  # Activation specification necessary for Conv2D and Dense layers

    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid'))  # 2nd conv. layer KEEP
    model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
    # model.add(Dropout(0.1))  # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Flatten())  # necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)

    # now start a typical neural network
    model.add(Dense(32))  # (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))

    # model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Dense(nb_classes))  # 10 final nodes (one for each class)  KEEP
    model.add(Activation('softmax'))  # softmax at end to pick between classes 0-9 KEEP

    # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
    # suggest you KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # and KEEP metrics at 'accuracy'
    # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # important inputs to the model: don't changes the ones marked KEEP
    batch_size = 20 # number of training samples used at a time to update the weights
    nb_classes = 3    # number of output possibilities: [0 - 2] KEEP
    nb_epoch = 1       # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols, channels = 300, 300, 4   # the size of the MNIST images KEEP
    input_shape = (img_rows, img_cols, channels)   # 1 channel image input (grayscale) KEEP
    nb_filters = 16  # number of convolutional filters to use
    pool_size = (2, 2)  # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (6, 6)  # convolutional kernel size, slides over image to learn features

    X_train, X_test, Y_train, Y_test = load_and_featurize_data()

    model = define_model(nb_filters, kernel_size, input_shape, pool_size)

    # during fit process watch train and test error simultaneously
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])  # this is the one we care about
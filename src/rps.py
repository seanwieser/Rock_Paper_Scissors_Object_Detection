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
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.transform import resize, rotate
import os
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

import numpy as np
np.random.seed(1337)  # for reproducibility


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print('Generating Train Images...')
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    print('Predicting Train')
    bottleneck_features_train = model.predict(
        generator, batch_size=nb_train_samples // batch_size, verbose=1)

    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)
    print('Train Bottleneck Features Saved...')

    print('Generating Test Images')
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print('Predicting Test')
    bottleneck_features_validation = model.predict(
        generator, batch_size=nb_validation_samples // batch_size, verbose=1)

    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    print('Test Bottleneck Features Saved...')


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_num, val_num = 612, 100
    train_labels = np.array([0]*train_num + [1]*train_num + [2]*train_num)

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0]*val_num + [1]*val_num + [2]*val_num)

    print('Building Model')
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)



if __name__ == '__main__':
    # dimensions of our images.
    img_width, img_height = 200, 300

    top_model_weights_path = '../data/bottleneck_fc_model.h5'
    train_data_dir = '../data/train'
    validation_data_dir = '../data/test'
    nb_train_samples = 612
    nb_validation_samples = 100
    epochs = 50
    batch_size = 16
    save_bottlebeck_features()
    train_top_model()



    # path to the model weights files.
    # weights_path = '../data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # top_model_weights_path = 'fc_model.h5'
    # dimensions of our images.
    # img_width, img_height = 150, 150

    # train_data_dir = 'cats_and_dogs_small/train'
    # validation_data_dir = 'cats_and_dogs_small/validation'
    # nb_train_samples = 2000
    # nb_validation_samples = 800
    # epochs = 50
    # batch_size = 16

    # # build the VGG16 network
    # model = applications.VGG16(weights='imagenet', include_top=False)
    # print('Model loaded.')

    # # build a classifier model to put on top of the convolutional model
    # top_model = Sequential()
    # top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # top_model.add(Dense(256, activation='relu'))
    # top_model.add(Dropout(0.5))
    # top_model.add(Dense(1, activation='sigmoid'))

    # # note that it is necessary to start with a fully-trained
    # # classifier, including the top classifier,
    # # in order to successfully do fine-tuning
    # top_model.load_weights(top_model_weights_path)

    # # add the model on top of the convolutional base
    # model.add(top_model)

    # # set the first 25 layers (up to the last conv block)
    # # to non-trainable (weights will not be updated)
    # for layer in model.layers[:25]:
    #     layer.trainable = False

    # # compile the model with a SGD/momentum optimizer
    # # and a very slow learning rate.
    # model.compile(loss='binary_crossentropy',
    #             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    #             metrics=['accuracy'])

    # # prepare data augmentation configuration
    # train_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True)

    # test_datagen = ImageDataGenerator(rescale=1. / 255)

    # train_generator = train_datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     class_mode='binary')

    # validation_generator = test_datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     class_mode='binary')

    # # fine-tune the model
    # model.fit_generator(
    #     train_generator,
    #     samples_per_epoch=nb_train_samples,
    #     epochs=epochs,
    #     validation_data=validation_generator,
    #     nb_val_samples=nb_validation_samples)






# def load_images(dir_name, data_kind):
#     img_lst = []
#     base_path = 'rps_images/' + data_kind
#     for filename in os.listdir(base_path + '/' + dir_name):
#         img_lst.append(io.imread(base_path + '/' + dir_name + '/' + filename))
#     return img_lst

# def load_and_featurize_data():
#     # train data 
#     paper_train = load_images('paper', 'train')
#     rock_train = load_images('rock', 'train')
#     sc_train= load_images('scissors', 'train')
#     X_train = np.array(paper_train + rock_train  + sc_train)

#     # test data
#     paper_test = load_images('paper', 'test')
#     rock_test = load_images('rock', 'test')
#     sc_test = load_images('scissors', 'test')
#     X_test = np.array(paper_test + rock_test  + sc_test)

#     # reshape input into format Conv2D layer likes
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)

#     # don't change conversion or normalization
#     X_train = X_train.astype('float32')  # data was uint8 [0-255]
#     X_test = X_test.astype('float32')    # data was uint8 [0-255]
#     X_train /= 255  # normalizing (scaling from 0 to 1)
#     X_test /= 255   # normalizing (scaling from 0 to 1)


#     print('X_train shape:', X_train.shape)
#     print(X_train.shape[0], 'train samples')
#     print(X_test.shape[0], 'test samples')

#     y_train = np.array([3*i//X_train.shape[0] for i in range(X_train.shape[0])])
#     y_test = np.array([3*i//X_test.shape[0] for i in range(X_test.shape[0])])

#     # convert class vectors to binary class matrices
#     Y_train = to_categorical(y_train, nb_classes)  # cool
#     Y_test = to_categorical(y_test, nb_classes)
    
#     # Shuffle all data
#     splits = [X_train, X_test, y_train, y_test]
#     rng_state = np.random.get_state()
#     for split in splits:
#         np.random.shuffle(split)
#         np.random.set_state(rng_state)


#     return X_train, X_test, Y_train, Y_test
# imports keras
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# import to read images
from skimage import io
from skimage import util 
import numpy as np
import os
from os import path
import matplotlib.pyplot as plt

def read_dataset_cnn(CLASS_NUM):
    # Load mnist data set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    WIDTH = x_train.shape[2]
    HEIGHT = x_train.shape[1]
    CLASS_NUM = 10

    # case by backend, 
    if K.image_data_format() == 'channels_first':
        #[channel, height, width]
        x_train = x_train.reshape(x_train.shape[0], 1, HEIGHT, WIDTH)
        x_test = x_test.reshape(x_test.shape[0], 1, HEIGHT, WIDTH)
        input_shape = (1, HEIGHT, WIDTH)
    else:
        #[height, width, channel]
        x_train = x_train.reshape(x_train.shape[0], HEIGHT, WIDTH, 1)
        x_test = x_test.reshape(x_test.shape[0], HEIGHT, WIDTH, 1)
        input_shape = (HEIGHT, WIDTH, 1)

    # image normalization
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class with one-hot encoding
    # 5 -> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    y_train = keras.utils.to_categorical(y_train, CLASS_NUM)
    y_test = keras.utils.to_categorical(y_test, CLASS_NUM)

    return x_train, x_test, y_train, y_test, input_shape

def read_dataset_dense(CLASS_NUM):
    # Load mnist data set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    WIDTH = x_train.shape[2]
    HEIGHT = x_train.shape[1]
    CLASS_NUM = 10

    #[height, width, channel]
    x_train = x_train.reshape(x_train.shape[0], HEIGHT*WIDTH)
    x_test = x_test.reshape(x_test.shape[0], HEIGHT*WIDTH)
    input_shape = [HEIGHT*WIDTH]

    # image normalization
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class with one-hot encoding
    # 5 -> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    y_train = keras.utils.to_categorical(y_train, CLASS_NUM)
    y_test = keras.utils.to_categorical(y_test, CLASS_NUM)

    return x_train, x_test, y_train, y_test, input_shape

def load_images_from_folder(folder):
    all_images = []
    set_list = os.listdir(folder)
    set_list.sort(key=lambda x: os.path.splitext(x)[0])
    for set_path in set_list:
        set_path_full = os.path.join(folder,set_path)
        image_list = os.listdir(set_path_full)
        image_list.sort(key=lambda x: int(os.path.splitext(x)[0]))
        for image_path in image_list:
            img = io.imread(os.path.join(set_path_full,image_path))
            img = util.invert(img)
            all_images.append(img)
    return np.array(all_images)

def read_dataset_image_cnn(CLASS_NUM):
    # Load mnist data set
    x_test_image = load_images_from_folder('test_images')
    y_test_image = []
    for i in range(int(x_test_image.shape[0]/CLASS_NUM)):
        y_test_image = y_test_image + list(range(CLASS_NUM))
    WIDTH = x_test_image.shape[2]
    HEIGHT = x_test_image.shape[1]

    # case by backend, 
    if K.image_data_format() == 'channels_first':
        #[channel, height, width]
        x_test_image = x_test_image.reshape(x_test_image.shape[0], 1, HEIGHT, WIDTH)
        input_shape = (1, HEIGHT, WIDTH)
    else:
        #[height, width, channel]
        x_test_image = x_test_image.reshape(x_test_image.shape[0], HEIGHT, WIDTH, 1)
        input_shape = (HEIGHT, WIDTH, 1)

    # image normalization
    x_test_image = x_test_image.astype('float32')
    x_test_image /= 255

    # convert class with one-hot encoding
    # 5 -> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    y_test_image = keras.utils.to_categorical(y_test_image, CLASS_NUM)

    return x_test_image, y_test_image, input_shape

def read_dataset_image_dense(CLASS_NUM):
    # Load mnist data set
    x_test_image = load_images_from_folder('test_images')
    y_test_image = []
    for i in range(int(x_test_image.shape[0]/CLASS_NUM)):
        y_test_image = y_test_image + list(range(CLASS_NUM))
    WIDTH = x_test_image.shape[2]
    HEIGHT = x_test_image.shape[1]

    x_test_image = x_test_image.reshape(x_test_image.shape[0], HEIGHT*WIDTH)
    input_shape = [HEIGHT*WIDTH]

    # image normalization
    x_test_image = x_test_image.astype('float32')
    x_test_image /= 255

    # convert class with one-hot encoding
    # 5 -> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    y_test_image = keras.utils.to_categorical(y_test_image, CLASS_NUM)

    return x_test_image, y_test_image, input_shape

def mnist_cnn_model(input_shape, CLASS_NUM):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(CLASS_NUM, activation='softmax'))

    return model

def mnist_cnn_dropout_model(input_shape, CLASS_NUM):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CLASS_NUM, activation='softmax'))

    return model

def mnist_dense_model(input_shape, CLASS_NUM):
    model = Sequential()
    model.add(Dense(256, activation='relu', 
                    input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(CLASS_NUM, activation='softmax'))

    return model

def mnist_dense_dropout_model(input_shape, CLASS_NUM):
    model = Sequential()
    model.add(Dense(256, activation='relu', 
                    input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CLASS_NUM, activation='softmax'))

    return model
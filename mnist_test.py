# imports keras
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils.vis_utils import plot_model

# Some cases, need GPU option because of memory size
# In my case, MX150, mobile GPU works with this option
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session=tf.Session(config=config)

# import test models
import mnist_test_model as mnist_test

# show images with matplot library
import numpy as np
import matplotlib.pyplot as plt

# Load mnist data set
CLASS_NUM = 10
EPOCH_NUM = 10
BATCH_NUM = 1000

# custom image dataset
(x_image_test_dense, y_image_test_dense, input_shape_test_dense) = mnist_test.read_dataset_image_dense(CLASS_NUM)
(x_image_test_cnn, y_image_test_cnn, input_shape_test_cnn) = mnist_test.read_dataset_image_cnn(CLASS_NUM)

# load from mnist dataset
(x_train_dense, x_test_dense, y_train_dense, y_test_dense, input_shape_dense) =  mnist_test.read_dataset_dense(CLASS_NUM)
(x_train_cnn, x_test_cnn, y_train_cnn, y_test_cnn, input_shape_cnn) =  mnist_test.read_dataset_cnn(CLASS_NUM)

# models list for test
model_list_dense = []
model_list_cnn = []
# mnist with fully connected network, drop out
model_list_dense_name = ['FC_nodropout', 'FC_dropout']
model_list_dense.append(mnist_test.mnist_dense_model(input_shape_dense, CLASS_NUM))
model_list_dense.append(mnist_test.mnist_dense_dropout_model(input_shape_dense, CLASS_NUM))
# mnist with CNN, drop out
model_list_cnn_name = ['CNN_nodropout', 'CNN_dropout']
model_list_cnn.append(mnist_test.mnist_cnn_model(input_shape_cnn, CLASS_NUM))
model_list_cnn.append(mnist_test.mnist_cnn_dropout_model(input_shape_cnn, CLASS_NUM))

# Learn FC and CNN

print('Test FC network and FC with drop out')
for i, model in enumerate(model_list_dense):
    # draw model, save
    plot_model(model, to_file='test_results/'+model_list_dense_name[i]+'_model.png', show_shapes=True, show_layer_names=True)
    
    with open('test_results/'+model_list_dense_name[i]+'_model.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    # Learning
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    history = model.fit(x_train_dense, y_train_dense,
                        batch_size=BATCH_NUM,
                        epochs=EPOCH_NUM,
                        verbose=1,
                        validation_data=(x_test_dense, y_test_dense))

    # draw loss, save
    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('test_results/'+model_list_dense_name[i]+'_loss.png')
    plt.cla()
    
    # evalutate with test image set
    score = model.evaluate(x_image_test_dense, y_image_test_dense, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('')

print('Test CNN and CNN with drop out')
for i, model in enumerate(model_list_cnn):
    # draw model, save
    plot_model(model, to_file='test_results/'+model_list_cnn_name[i]+'_model.png', show_shapes=True, show_layer_names=True)

    with open('test_results/'+model_list_cnn_name[i]+'_model.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    # Learning
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    history = model.fit(x_train_cnn, y_train_cnn,
                        batch_size=BATCH_NUM,
                        epochs=EPOCH_NUM,
                        verbose=1,
                        validation_data=(x_test_cnn, y_test_cnn))

    # draw loss, save
    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('test_results/'+model_list_cnn_name[i]+'_loss.png')
    plt.cla()

    # evalutate with test image set
    score = model.evaluate(x_image_test_cnn, y_image_test_cnn, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('')

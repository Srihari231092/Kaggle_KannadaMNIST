# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

import keras
from keras.layers import Convolution2D, BatchNormalization, merge, Cropping2D, concatenate

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


smooth = 1e-12

num_mask_channels = 1


def cnn_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='tanh', input_shape=input_shape))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='tanh'))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='tanh'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(units=128, activation='tanh'))

    model.add(Dropout(0.25))

    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = cnn_model(input_shape=(28, 28, 1), num_classes=10)
    print(model.summary())
    plot_model(model, to_file='./models/cnn_model.png', show_shapes=True)

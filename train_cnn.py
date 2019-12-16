from cnn_model import *
import pandas as pd
import os
import time

from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from argparse import ArgumentParser

from sklearn.model_selection import train_test_split


def main(train_path, num_classes, batch_size, n_epochs, input_weights_path, output_weights_path, logfile_path):

    # Read the input data
    train = pd.read_csv(train_path)

    # input image dimensions
    img_rows, img_cols = 28, 28

    train_data, val_data = train_test_split(train, test_size=0.25)
    x_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    x_val = val_data.iloc[:, 1:].values
    y_val = val_data.iloc[:, 0].values

    if K.image_data_format() == 'channels_first':
        print("Channels first --------")
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train /= 255
    x_val /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model = cnn_model(input_shape=(input_shape), num_classes=num_classes)

    if os.path.isfile(input_weights_path):
        model.load_weights(input_weights_path)

    model_checkpoint = ModelCheckpoint(output_weights_path,
                                       monitor='val_loss',
                                       save_best_only=True)
    # Logger configuration
    csv_logger = CSVLogger(logfile_path, append=True, separator=';')
    # Tensorboard logging configuration
    tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)

    # Fit the model using ImageDataAugmentor from Keras
    # Keras documentation - https://keras.io/preprocessing/image/
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        zoom_range=[0.9,1.1],
        vertical_flip=False)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size,
                                     seed=int(time.time())),  # Change the random seed every time
                        verbose=2,
                        shuffle=True,
                        callbacks=[model_checkpoint, csv_logger, tensorboard],
                        validation_data=(x_val, y_val),
                        epochs=n_epochs)

    score = model.evaluate(x_val, y_val, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':

    parser = ArgumentParser()

    # train file path
    parser.add_argument("-t", "--train_path", dest="train_path",
                        help="file path for training data [.csv]", metavar="./PATH", required=True)

    # log file path
    parser.add_argument("-l", "--log_path", dest="log_path",
                        help="file path for training data [.csv]", metavar="./PATH", required=True)

    # input weights file path
    parser.add_argument("-i", "--input_weights_path", dest="input_weights_path",
                        help="file path for input weights", metavar="./PATH", required=True)

    # output weights file path
    parser.add_argument("-o", "--output_weights_path", dest="output_weights_path",
                        help="file path for output weights", metavar="./PATH", required=True)

    # num_epochs
    parser.add_argument("-e", "--num_epochs", dest="num_epochs",
                        help="number of epochs", metavar="100", required=True)

    # batch size
    parser.add_argument("-b", "--batch_size", dest="batch_size",
                        help="Batch size", metavar="200", required=True)

    # number of classes
    parser.add_argument("-n", "--num_classes", dest="num_classes",
                        help="Number of classes", metavar="10", required=True)

    args = parser.parse_args()

    for arg in vars(args):
        print(arg, "-->", getattr(args, arg))

    main(train_path=args.train_path,
         num_classes=int(args.num_classes),
         batch_size=int(args.batch_size),
         n_epochs=int(args.num_epochs),
         input_weights_path=args.input_weights_path,
         output_weights_path=args.output_weights_path,
         logfile_path=args.log_path
         )

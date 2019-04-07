import scipy.io
import numpy as np
import tensorflow as tf
import scipy.misc
import cv2

mnist = tf.keras.datasets.mnist


def prepare_svhn_data(train_path, test_path, valid_split=0.1):
    train_data = scipy.io.loadmat(train_path)
    test_data = scipy.io.loadmat(test_path)

    # cv2.imwrite('name.png', cv2.cvtColor(train_data['X'][:,:,:,0], cv2.COLOR_RGB2BGR))

    axes = (3, 0, 1, 2)

    x_train = np.transpose(train_data['X'], axes)
    y_train = train_data['y'].flatten()

    # cv2.imwrite('name.png', cv2.cvtColor(x_train[0], cv2.COLOR_RGB2BGR))

    l_train_x = len(x_train)
    x_train, x_valid = np.split(x_train, [l_train_x - int(valid_split * l_train_x)])
    y_train, y_valid = np.split(y_train, [l_train_x - int(valid_split * l_train_x)])

    x_test = np.transpose(test_data['X'], axes)
    y_test = test_data['y'].flatten()

    y_train = y_train - 1
    y_valid = y_valid - 1
    y_test = y_test - 1

    return (x_train / 255, y_train), (x_valid / 255, y_valid), (x_test / 255, y_test)


def prepare_mnist_data(valid_split=0.1):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    l_train_x = len(x_train)
    x_train, x_valid = np.split(x_train, [l_train_x - int(valid_split * l_train_x)])
    y_train, y_valid = np.split(y_train, [l_train_x - int(valid_split * l_train_x)])

    out_shape = (-1, 28, 28, 1)
    x_train = x_train.reshape(out_shape)
    x_valid = x_valid.reshape(out_shape)
    x_test = x_test.reshape(out_shape)

    return (x_train / 255, y_train), (x_valid / 255, y_valid), (x_test / 255, y_test)

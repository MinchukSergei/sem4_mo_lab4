import tensorflow as tf
import math
from pathlib import Path
import re

ACCURACY_PATTERN = r'(?<=weights\.\d{2}-)(.*)(?=\.hdf5)'
ROOT_PATH = 'D:/Programming/bsuir/sem4/MO/lab4'


class LeNet5:
    def __init__(self, in_shape, out_shape, data, f_act=tf.nn.relu, epochs=30, use_mnist=False):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.data = data
        self.f_act = f_act
        self.epochs = epochs
        self.model = None
        self.use_mnist = use_mnist

    def build_model(self):
        model = self.load_model()

        if not model:
            self.init_model()
            self.fit_model()
        else:
            self.model = model

    def predict(self, img):
        img = img.reshape(-1, *self.in_shape)

        return self.model.predict_classes(img) + (0 if self.use_mnist else 1)

    def load_model(self):
        models = Path(f'{ROOT_PATH}/models').glob('*.hdf5')
        best_acc = 0
        best_model = None

        for m in list(models):
            acc = float(re.search(ACCURACY_PATTERN, m.name).group())
            if acc > best_acc:
                best_acc = acc
                best_model = m

        if best_model:
            best_model = tf.keras.models.load_model(best_model)

        return best_model

    def init_model(self):
        model = tf.keras.models.Sequential()

        model.add(
            tf.keras.layers.Conv2D(
                6,
                kernel_size=(5, 5),
                kernel_regularizer='l2',
                strides=(1, 1),
                activation=self.f_act,
                input_shape=self.in_shape,
                padding='same' if self.use_mnist else 'valid'
            )
        )
        model.add(
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2)
            )
        )
        model.add(
            tf.keras.layers.Conv2D(
                16,
                kernel_size=(5, 5),
                kernel_regularizer='l2',
                strides=(1, 1),
                activation=self.f_act
            )
        )
        model.add(
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2)
            )
        )
        model.add(tf.keras.layers.Flatten())
        model.add(
            tf.keras.layers.Dense(
                84,
                activation=self.f_act,
                kernel_regularizer='l2'
            )
        )
        model.add(
            tf.keras.layers.Dense(
                self.out_shape,
                activation='softmax'
            )
        )

        model.compile(
            optimizer=tf.keras.optimizers.SGD(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

    def fit_model(self):
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = self.data

        history = LossHistory()
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
        save_model = tf.keras.callbacks.ModelCheckpoint(ROOT_PATH + '/models/weights.{epoch:02d}-{val_acc:.4f}.hdf5',
                                                        save_best_only=True)

        self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            validation_data=(x_valid, y_valid),
            callbacks=[history, lrate, save_model],
            use_multiprocessing=True
        )

        self.model.evaluate(x_test, y_test)


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))


def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 5.0

    return initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

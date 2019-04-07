import tensorflow as tf
import math

mnist = tf.keras.datasets.mnist


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
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(lrate)
    return lrate


def fc_nn(in_shape, out_shape, data, f_act=tf.nn.relu, epochs=30):
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), kernel_regularizer='l2', strides=(1, 1), activation=f_act, input_shape=in_shape))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), kernel_regularizer='l2', strides=(1, 1), activation=f_act))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(84, activation=f_act, kernel_regularizer='l2'))
    model.add(tf.keras.layers.Dense(out_shape, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = LossHistory()
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid, y_valid), callbacks=[history, lrate], use_multiprocessing=True)

    model.evaluate(x_test, y_test)

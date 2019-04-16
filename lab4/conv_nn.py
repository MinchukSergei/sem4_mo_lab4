import numpy as np

import tensorflow as tf


class ConvNN:
    def __init__(self, in_shape, data, epochs=30, use_prev=False):
        self.in_shape = in_shape
        self.data = data
        self.epochs = epochs
        self.use_prev = use_prev

    def build_model(self):
        self.init_model()
        self.fit_model()
        self.test_model()

    def predict(self, img):
        fetches = [self.logits_1, self.logits_2, self.logits_3, self.logits_4, self.logits_5]

        feed_dict_batch = {
            self.x: [img],
            self.drop_rate: 0
        }

        l1, l2, l3, l4, l5 = self.session.run(fetches, feed_dict=feed_dict_batch)
        digits_pred = predictions(l1, l2, l3, l4, l5)

        return np.array2string(digits_pred[digits_pred < 10], separator='')

    def init_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.placeholder(tf.int32, shape=[None, 6])
        self.drop_rate = tf.placeholder(tf.float32)

        filters1 = 16
        h_conv1 = conv_2d(self.x, weights("WC_1", [5, 5, 3, filters1])) + bias("BC_1", [filters1])
        h_conv1 = tf.layers.batch_normalization(h_conv1, training=True)
        h_conv1 = tf.nn.relu(h_conv1)
        h_conv1 = max_pool(h_conv1, 2)
        h_conv1 = tf.nn.dropout(h_conv1, rate=self.drop_rate)

        filters2 = 24
        h_conv2 = conv_2d(h_conv1, weights("WC_2", [5, 5, filters1, filters2])) + bias("BC_2", [filters2])
        h_conv2 = tf.layers.batch_normalization(h_conv2, training=True)
        h_conv2 = tf.nn.relu(h_conv2)
        h_conv2 = tf.nn.dropout(h_conv2, rate=self.drop_rate)

        filters3 = 32
        h_conv3 = conv_2d(h_conv2, weights("WC_3", [5, 5, filters2, filters3])) + bias("BC_3", [filters3])
        h_conv3 = tf.layers.batch_normalization(h_conv3, training=True)
        h_conv3 = tf.nn.relu(h_conv3)
        h_conv3 = max_pool(h_conv3, 2)
        h_conv3 = tf.nn.dropout(h_conv3, rate=self.drop_rate)

        filters4 = 48
        h_conv4 = conv_2d(h_conv3, weights("WC_4", [5, 5, filters3, filters4])) + bias("BC_4", [filters4])
        h_conv4 = tf.layers.batch_normalization(h_conv4, training=True)
        h_conv4 = tf.nn.relu(h_conv4)
        h_conv4 = tf.nn.dropout(h_conv4, rate=self.drop_rate)

        filters5 = 64
        h_conv5 = conv_2d(h_conv4, weights("WC_5", [5, 5, filters4, filters5])) + bias("BC_5", [filters5])
        h_conv5 = tf.layers.batch_normalization(h_conv5, training=True)
        h_conv5 = tf.nn.relu(h_conv5)
        h_conv5 = max_pool(h_conv5, 2)
        h_conv5 = tf.nn.dropout(h_conv5, rate=self.drop_rate)

        filters6 = 64
        h_conv6 = conv_2d(h_conv5, weights("WC_6", [5, 5, filters5, filters6])) + bias("BC_6", [filters6])
        h_conv6 = tf.layers.batch_normalization(h_conv6, training=True)
        h_conv6 = tf.nn.relu(h_conv6)
        h_conv6 = tf.nn.dropout(h_conv6, rate=self.drop_rate)

        filters7 = 64
        h_conv7 = conv_2d(h_conv6, weights("WC_7", [5, 5, filters6, filters7])) + bias("BC_7", [filters7])
        h_conv7 = tf.layers.batch_normalization(h_conv7, training=True)
        h_conv7 = tf.nn.relu(h_conv7)
        h_conv7 = tf.nn.dropout(h_conv7, rate=self.drop_rate)

        filters8 = 64
        h_conv8 = conv_2d(h_conv7, weights("WC_8", [5, 5, filters7, filters8])) + bias("BC_8", [filters8])
        h_conv8 = tf.layers.batch_normalization(h_conv8, training=True)
        h_conv8 = tf.nn.relu(h_conv8)
        h_conv8 = tf.nn.dropout(h_conv8, rate=self.drop_rate)

        h_flat = tf.reshape(h_conv8, [-1, 4 * 4 * filters8])

        units1 = 2048
        h_fc1 = tf.nn.relu(
            tf.matmul(h_flat, weights("WFC_1", [4 * 4 * filters8, units1])) + bias("BFC_1", [units1]))
        h_fc1 = tf.nn.dropout(h_fc1, rate=self.drop_rate)

        units2 = 2048
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, weights("WFC_2", [units1, units2])) + bias("BFC_2", [units2]))
        h_fc2 = tf.nn.dropout(h_fc2, rate=self.drop_rate)

        units3 = 2048
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, weights("WFC_3", [units2, units3])) + bias("BFC_3", [units3]))
        h_fc3 = tf.nn.dropout(h_fc3, rate=self.drop_rate)

        digit_len = 11
        W1 = tf.get_variable(shape=[units2, digit_len], name="WFC1",
                             initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable(shape=[units2, digit_len], name="WFC2",
                             initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable(shape=[units2, digit_len], name="WFC3",
                             initializer=tf.contrib.layers.xavier_initializer())
        W4 = tf.get_variable(shape=[units2, digit_len], name="WFC4",
                             initializer=tf.contrib.layers.xavier_initializer())
        W5 = tf.get_variable(shape=[units2, digit_len], name="WFC5",
                             initializer=tf.contrib.layers.xavier_initializer())

        b1 = bias("BFC1", [11])
        b2 = bias("BFC2", [11])
        b3 = bias("BFC3", [11])
        b4 = bias("BFC4", [11])
        b5 = bias("BFC5", [11])

        self.logits_1 = tf.matmul(h_fc3, W1) + b1
        self.logits_2 = tf.matmul(h_fc3, W2) + b2
        self.logits_3 = tf.matmul(h_fc3, W3) + b3
        self.logits_4 = tf.matmul(h_fc3, W4) + b4
        self.logits_5 = tf.matmul(h_fc3, W5) + b5

        l2_reg = 0.001
        wc1 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WC_1:0')) * l2_reg
        wc2 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WC_2:0')) * l2_reg
        wc3 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WC_3:0')) * l2_reg
        wc4 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WC_4:0')) * l2_reg
        wc5 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WC_5:0')) * l2_reg
        wc6 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WC_6:0')) * l2_reg
        wc7 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WC_7:0')) * l2_reg
        wc8 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WC_7:0')) * l2_reg

        wfc1 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WFC_1:0')) * l2_reg
        wfc2 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WFC_2:0')) * l2_reg
        wfc3 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WFC_3:0')) * l2_reg

        wo1 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WFC1:0')) * l2_reg
        wo2 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WFC2:0')) * l2_reg
        wo3 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WFC3:0')) * l2_reg
        wo4 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WFC4:0')) * l2_reg
        wo5 = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('WFC5:0')) * l2_reg

        regularizer = wc1 + wc2 + wc3 + wc4 + wc5 + wc6 + wc7 + wc8 + wfc1 + wfc2 + wfc3 + wo1 + wo2 + wo3 + wo4 + wo5

        loss1 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_1, labels=self.y[:, 1]))
        loss2 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_2, labels=self.y[:, 2]))
        loss3 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_3, labels=self.y[:, 3]))
        loss4 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_4, labels=self.y[:, 4]))
        loss5 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_5, labels=self.y[:, 5]))

        self.loss = loss1 + loss2 + loss3 + loss4 + loss5 + regularizer

        global_step = tf.Variable(0)
        # 0.97 0.01
        learning_rate = tf.train.exponential_decay(0.01, global_step, len(self.data[0][0]), 0.98)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss,
                                                                                   global_step=global_step)
        self.saver = tf.train.Saver()

    def fit_model(self):
        batch_size = 256

        (x_train, y_train), (x_valid, y_valid) = self.data[:-1]

        num_train_iter = int(len(x_train) / batch_size)
        num_valid_iter = int(len(y_valid) / batch_size)

        best_acc = 0

        self.init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(self.init)

        if self.use_prev:
            self.saver.restore(self.session, 'D:/Programming/bsuir/sem4/MO/lab4/models/model0.724.ckpt')
            return

        for epoch in range(self.epochs):
            print('Training epoch: {}'.format(epoch + 1))

            tr_x, tr_y = data_shuffle(x_train, y_train)
            v_x, v_y = data_shuffle(x_valid, y_valid)

            for iteration in range(num_train_iter):
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(tr_x, tr_y, start, end)

                feed_dict_batch = {
                    self.x: x_batch,
                    self.y: y_batch,
                    self.drop_rate: 0.1  # 0.15
                }

                _, loss = self.session.run([self.optimizer, self.loss], feed_dict=feed_dict_batch)

                if iteration % 10 == 0:
                    fetches = [self.logits_1, self.logits_2, self.logits_3, self.logits_4, self.logits_5, self.y]

                    feed_dict_batch = {
                        self.x: x_batch,
                        self.y: y_batch,
                        self.drop_rate: 0
                    }

                    l1, l2, l3, l4, l5, expected_labels = self.session.run(fetches, feed_dict=feed_dict_batch)
                    acc = accuracy(l1, l2, l3, l4, l5, expected_labels)
                    print("iter {0:3d}:\t TRAIN Loss={1:.2f},\t Accuracy={2:.01%}".format(iteration, loss, acc))

            acc_valid = 0
            for iteration in range(num_valid_iter):
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(v_x, v_y, start, end)

                feed_dict_batch = {
                    self.x: x_batch,
                    self.y: y_batch,
                    self.drop_rate: 0
                }

                fetches = [self.logits_1, self.logits_2, self.logits_3, self.logits_4, self.logits_5, self.y]
                l1, l2, l3, l4, l5, expected_labels = self.session.run(fetches, feed_dict=feed_dict_batch)

                acc_valid += accuracy(l1, l2, l3, l4, l5, expected_labels)

            acc_valid = acc_valid / num_valid_iter

            if acc_valid > best_acc:
                best_acc = acc_valid
                self.saver.save(self.session, f'./models/model{best_acc:.3f}.ckpt')

            print('---------------------------------------------------------')
            print("Epoch: {0}:\t VALID Accuracy={1:.01%}".format(epoch + 1, acc_valid))
            print('---------------------------------------------------------')

    def test_model(self):
        (x_test, y_test) = self.data[2]

        batch_size = 256
        num_test_iter = int(len(x_test) / batch_size)

        acc_valid = 0
        for iteration in range(num_test_iter):
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_test, y_test, start, end)

            feed_dict_batch = {
                self.x: x_batch,
                self.y: y_batch,
                self.drop_rate: 0
            }

            fetches = [self.logits_1, self.logits_2, self.logits_3, self.logits_4, self.logits_5, self.y]
            l1, l2, l3, l4, l5, pred_labels = self.session.run(fetches, feed_dict=feed_dict_batch)

            acc_valid += accuracy(l1, l2, l3, l4, l5, pred_labels)

        print('---------------------------------------------------------')
        print("TEST Accuracy={0:.01%}".format(acc_valid / num_test_iter))
        print('---------------------------------------------------------')


def predictions(logit_1, logit_2, logit_3, logit_4, logit_5):
    digit1 = np.argmax(logit_1, axis=1)
    digit2 = np.argmax(logit_2, axis=1)
    digit3 = np.argmax(logit_3, axis=1)
    digit4 = np.argmax(logit_4, axis=1)
    digit5 = np.argmax(logit_5, axis=1)
    return np.stack([digit1, digit2, digit3, digit4, digit5], axis=1)


def accuracy(logit_1, logit_2, logit_3, logit_4, logit_5, labels):
    labels = labels[:, 1:]

    digits = predictions(logit_1, logit_2, logit_3, logit_4, logit_5)

    total = len(labels)
    correct = 0
    for i in range(len(labels)):
        if np.array_equal(digits[i], labels[i]):
            correct += 1

    return correct / total


def get_next_batch(data_x, data_y, start, end):
    x_batch = data_x[start:end]
    y_batch = data_y[start:end]

    return x_batch, y_batch


def data_shuffle(x, y):
    permutation = np.random.permutation(y.shape[0])

    x_shuffled = x[permutation, :]
    y_shuffled = y[permutation, :]

    return x_shuffled, y_shuffled


def bias(name, shape):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, stride):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME')


def weights(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

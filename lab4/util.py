import cv2
import tensorflow as tf

from lab4.data_reader import prepare_svhn_data
from lab4.lenet5 import LeNet5


class Util:
    def __init__(self):
        self.model = None

    def build_model(self):
        lenet5 = LeNet5(
            (32, 32, 3),
            10,
            prepare_svhn_data(),
            epochs=30
        )

        lenet5.build_model()

        self.model = lenet5

    def predict(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (32, 32))
        img = img / 255

        return self.model.predict(img)

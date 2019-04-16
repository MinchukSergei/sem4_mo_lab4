import cv2

from lab4.data_reader import prepare_svhn_data
from lab4.conv_nn import ConvNN


class Util:
    def __init__(self):
        self.model = None

    def build_model(self):
        lenet5 = ConvNN(
            (32, 32, 3),
            prepare_svhn_data(),
            epochs=30,
            use_prev=True
        )

        lenet5.build_model()

        self.model = lenet5

    def predict(self, path):
        img = cv2.imread(path)

        imgs = [img]

        digits = []
        for im in imgs:
            im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
            im = cv2.dilate(im, (3, 3))
            im = im / 255
            digits.append(self.model.predict(im))

        return digits

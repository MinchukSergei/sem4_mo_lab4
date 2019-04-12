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

        # imgs = self.split_image(img)
        imgs = [img]

        digits = []
        for im in imgs:
            im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
            im = cv2.dilate(im, (3, 3))
            im = im / 255
            digits.append(self.model.predict(im))

        return digits

    def split_image(self, img):
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
        ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        imgs = []
        rects.sort(key=lambda a: a[0])

        for rect in rects:
            leng = int(rect[3] * 1.6)
            pt1 = max(int(rect[1] + rect[3] // 2 - leng // 2), 0)
            pt2 = max(int(rect[0] + rect[2] // 2 - leng // 2), 0)
            roi = img[pt1:pt1 + leng, pt2:pt2 + leng]

            imgs.append(roi)

        return imgs
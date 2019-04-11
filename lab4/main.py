# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from lab4.lenet5 import LeNet5
from lab4.data_reader import prepare_mnist_data, prepare_svhn_data
import cv2
from pathlib import Path

USE_MNIST = False


def main():
    if USE_MNIST:
        lenet5 = LeNet5(
            (28, 28, 1),
            10,
            prepare_mnist_data(),
            epochs=30,
            use_mnist=USE_MNIST
        )
    else:
        lenet5 = LeNet5(
            (32, 32, 3),
            10,
            prepare_svhn_data(),
            epochs=30,
            use_mnist=USE_MNIST
        )

    lenet5.build_model()

    test(lenet5)


def test(lenet5):
    x_tests = Path('./test').glob('*.jpg')
    y_test = [1, 1, 2, 2, 3, 4, 7, 9]

    for i, t in enumerate(x_tests):
        im = cv2.imread(str(t.absolute()))
        im = cv2.resize(im, (32, 32))
        im = im / 255

        cl = lenet5.predict(im)
        print(f'Predicted: {cl}. Expected: {y_test[i]}')


if __name__ == '__main__':
    main()

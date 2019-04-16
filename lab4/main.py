# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from lab4.conv_nn import ConvNN
from lab4.data_reader import prepare_svhn_data
import cv2
from pathlib import Path


def main():
    conv_nn = ConvNN(
        (32, 32, 3),
        prepare_svhn_data(),
        epochs=150,
        use_prev=True
    )

    conv_nn.build_model()

    test(conv_nn)


def test(lenet5):
    x_tests = Path('./test/test').glob('*.png')
    y_test = [26, 34, 37, 49, 189]

    for i, t in enumerate(x_tests):
        im = cv2.imread(str(t.absolute()))
        im = cv2.resize(im, (32, 32))
        im = im / 255

        cl = lenet5.predict(im)
        print(f'Predicted: {cl}. Expected: {y_test[i]}')


if __name__ == '__main__':
    main()

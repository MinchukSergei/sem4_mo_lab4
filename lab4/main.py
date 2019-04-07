# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
from lab4.fc_nn import fc_nn
from lab4.data_reader import prepare_mnist_data, prepare_svhn_data


def main():
    path_prefix = Path('D:\\Programming\\bsuir\\sem4\\MO\\data\\lab4\\')
    train_path = path_prefix / 'train_32x32.mat'
    test_path = path_prefix / 'test_32x32.mat'

    # fc_nn([512, 512, 512], (28, 28, 1), 10, prepare_mnist_data(), [0.1, 0.2, 0.3])

    fc_nn(
        (32, 32, 3),
        10,
        prepare_svhn_data(train_path, test_path),
        epochs=30
    )


if __name__ == '__main__':
    main()

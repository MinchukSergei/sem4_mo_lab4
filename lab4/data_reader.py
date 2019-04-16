from pathlib import Path

import numpy as np
import h5py
import csv
import cv2
import math
from tqdm import tqdm


def convert_to_csv(r, w):
    digit_struct = r['digitStruct']
    names = digit_struct['name']
    bboxes = digit_struct['bbox']
    csv_writer = csv.writer(w, delimiter=',')

    csv_writer.writerow(['name', 'label', 'left', 'top', 'width', 'height'])

    for n, bb in tqdm(zip(names, bboxes)):
        name = ''.join(chr(i) for i in r[n[0]])

        md = dict()
        md['label'] = []
        md['left'] = []
        md['top'] = []
        md['width'] = []
        md['height'] = []

        def fill_bbox_data(name, obj):
            vals = []
            if obj.shape[0] == 1:
                vals.append(obj[0][0])
            else:
                for k in range(obj.shape[0]):
                    vals.append(int(r[obj[k][0]][0][0]))
            md[name] = vals

        r[bb[0]].visititems(fill_bbox_data)

        for label, left, top, width, height in zip(md['label'], md['left'], md['top'], md['width'], md['height']):
            csv_writer.writerow([name, label, left, top, width, height])

def preprocess_data(path_csv, images_path):
    csv_data = []

    with open(path_csv) as f:
        csv_reader = list(csv.reader(f, delimiter=','))

        for row in csv_reader[1:]:
            csv_data.append((row[0], int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4])), int(float(row[5]))))

    imgs_data = {}
    for im_data in csv_data:
        im_name = im_data[0]

        if im_name not in imgs_data:
            imgs_data[im_name] = []

        imgs_data[im_name].append((im_data[1], im_data[2], im_data[3], im_data[4], im_data[5]))

    images = np.empty((len(imgs_data), 32, 32, 3))
    labels = np.empty((len(imgs_data), 6))

    for idx, (n, boxes) in tqdm(enumerate(imgs_data.items())):
        im_path = images_path / n

        min_l = math.inf
        min_t = math.inf
        max_r = 0
        max_d = 0
        im_labels = [10, 10, 10, 10, 10]

        for i, b in enumerate(boxes):
            if i > 4:
                continue
            im_labels[i] = b[0]
            left = b[1]
            top = b[2]
            right = left + b[3]
            down = top + b[4]
            min_l = min(left, min_l)
            min_t = min(top, min_t)
            max_r = max(right, max_r)
            max_d = max(down, max_d)

        im_labels.insert(0, min(5, len(boxes)))

        img = cv2.imread(str(im_path.absolute()))
        cropped = img[max(min_t, 0):min(max_d, img.shape[0]), max(min_l, 0):max(max_r, img.shape[1])]
        resized = cv2.resize(cropped, (32, 32))
        images[idx] = resized / 255
        labels[idx] = im_labels

    return images, labels


def prepare_svhn_data(valid_split=0.1):
    mat_prefix = Path('D:/Programming/bsuir/sem4/MO/data/lab4')
    csv_prefix = Path('D:/Programming/bsuir/sem4/MO/lab4/data')

    train_path = mat_prefix / 'train' / 'digitStruct.mat'
    test_path = mat_prefix / 'test' / 'digitStruct.mat'
    train_path_csv = csv_prefix / 'train.csv'
    test_path_csv = csv_prefix / 'test.csv'

    if not train_path_csv.exists():
        with h5py.File(str(train_path.absolute())) as r, open(train_path_csv, 'w+', newline='') as w:
            convert_to_csv(r, w)

    if not test_path_csv.exists():
        with h5py.File(str(test_path.absolute())) as r, open(test_path_csv, 'w+', newline='') as w:
            convert_to_csv(r, w)

    train_npz = csv_prefix / 'train32.npz'
    test_npz = csv_prefix / 'test32.npz'

    if train_npz.exists():
        train = np.load(train_npz)
        x_train = train['x_train']
        y_train = train['y_train']
    else:
        x_train, y_train = preprocess_data(train_path_csv, mat_prefix / 'train')
        np.savez(train_npz, x_train=x_train, y_train=y_train)

    if test_npz.exists():
        test = np.load(test_npz)
        x_test = test['x_test']
        y_test = test['y_test']
    else:
        x_test, y_test = preprocess_data(test_path_csv, mat_prefix / 'test')
        np.savez(test_npz, x_test=x_test, y_test=y_test)

    l_train_x = len(x_train)
    x_train, x_valid = np.split(x_train, [l_train_x - int(valid_split * l_train_x)])
    y_train, y_valid = np.split(y_train, [l_train_x - int(valid_split * l_train_x)])

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

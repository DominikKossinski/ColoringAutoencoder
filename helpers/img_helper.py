from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tf_ds
from PIL import Image
from tensorflow.image import rgb_to_hsv
from tqdm import tqdm


def load_image(path, size: Tuple[int, int] = None, is_hsv: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(path)
    if size is not None:
        img = img.resize(size)
    gray_scale = np.mean(np.asarray(img, dtype='float32') / 255.0, axis=2)
    gray_scale = np.reshape(gray_scale, (gray_scale.shape[0], gray_scale.shape[1], 1))
    img = np.asarray(img, dtype='float32') / 255.0
    if is_hsv:
        img = rgb_to_hsv(img)
    return img, gray_scale


def load_data(name, nb_samples, size=100, is_hsv: bool = False):
    size = (size, size)
    train = tf_ds.load(name=name, split=tf_ds.Split.TEST)
    train = train.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)

    train_x = prepare_data(train, "Train dataset", nb_samples, size, is_hsv)

    val = tf_ds.load(name=name, split=tf_ds.Split.TRAIN)
    val = val.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)

    val_x = prepare_data(val, "Validation dataset", nb_samples // 5, size, is_hsv)
    return train_x, val_x


def prepare_data(dataset, name, n, size, is_hsv: bool):
    samples = list(dataset)[:n]
    data_x = []
    # data_y = []
    i = 0
    for element in tqdm(samples, desc=name):
        img = Image.fromarray(np.array(element["image"]))
        if size is not None:
            img = img.resize(size)
        # gray_scale = img.convert('L')
        # x = np.mean(np.asarray(gray_scale) / 255.0, axis=2, dtype="float32")
        y = np.asarray(img, dtype="float32") / 255.0
        # if is_hsv:
        #     y = rgb_to_hsv(y)
        data_x.append(y)
        if i == 0:
            plt.imshow(y)
            plt.show()
            i += 1
        # data_y.append(y)
    data_x = np.array(data_x, dtype='float32')
    # data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], data_x.shape[2], 3))
    # data_y = np.array(data_y, dtype='float32')
    return data_x  # , data_y


class AutoEncoderImageDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):

    def __init__(self, is_hsv: bool):
        super(AutoEncoderImageDataGenerator, self).__init__(
            rotation_range=30,
            width_shift_range=1.0,
            height_shift_range=1.0,
            horizontal_flip=True,
            vertical_flip=True,
        )
        self.__is_hsv = is_hsv

    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):
        steps = len(x) // batch_size
        x = super(AutoEncoderImageDataGenerator, self).flow(
            x,
            y=None,
            batch_size=batch_size,
            shuffle=True,
            sample_weight=None,
            seed=None,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            subset=None)
        print(x)
        x_data = []
        y_data = []
        n = 0
        for batch in tqdm(x):
            if n >= steps:
                break
            # x_batch = []
            # y_batch = []
            for i in batch:
                x_data.append(tf.math.reduce_mean(i / 255.0, axis=2, keepdims=True))
                y_data.append(i if not self.__is_hsv else rgb_to_hsv(i))
            # x_data.append(x_batch)
            # y_data.append(y_batch)
            n += 1
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        print(x_data.shape)
        print(y_data.shape)
        # if self.__is_hsv:
        #     y = rgb_to_hsv(x)
        # else:
        #     y = x
        # print(tf.math.reduce_mean(x/255.0, axis=2))
        # print(y)
        # plt.imshow(x_data[0], cmap='gray')
        # print(x_data[0])
        # plt.show()
        #
        # plt.imshow(y_data[0])
        # plt.show()
        return tf.keras.preprocessing.image.NumpyArrayIterator(x_data, y_data, self, batch_size)
        # return tf.math.reduce_mean(x / 255.0, axis=2), y

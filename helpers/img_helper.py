from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tf_ds
from PIL import Image
from tensorflow.image import rgb_to_hsv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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


def load_data(name, nb_samples, size=100):
    size = (size, size)
    train = tf_ds.load(name=name, split=tf_ds.Split.TEST)
    train = train.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)

    train_x = prepare_data(train, "Train dataset", nb_samples, size)

    val = tf_ds.load(name=name, split=tf_ds.Split.TRAIN)
    val = val.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)

    val_x = prepare_data(val, "Validation dataset", nb_samples // 5, size)
    return train_x, val_x


def prepare_data(dataset, name, n, size):
    samples = list(dataset)[:n]
    data_x = []
    for element in tqdm(samples, desc=name):
        img = Image.fromarray(np.array(element["image"]))
        if size is not None:
            img = img.resize(size)
        x = np.asarray(img, dtype="float32") / 255.0
        data_x.append(x)
    data_x = np.array(data_x, dtype='float32')
    return data_x


class AutoEncoderImageDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x_train, batch_size: int, shuffle: bool, is_hsv: bool):
        super(AutoEncoderImageDataGenerator, self).__init__()
        self.__image_data_generator = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=1.0,
            height_shift_range=1.0,
            horizontal_flip=True,
            vertical_flip=True,
        )
        self.__x_train = x_train
        self.__batch_size: int = batch_size
        self.__shuffle: bool = shuffle
        self.__is_hsv: bool = is_hsv

    def __getitem__(self, index):
        batch = self.__x_train[index * self.__batch_size: (index + 1) * self.__batch_size]
        return self.__prepare_data(batch)

    def __len__(self):
        return len(self.__x_train) // self.__batch_size

    def on_epoch_end(self):
        if self.__shuffle:
            self.__x_train = np.random.shuffle(self.__x_train)

    def __prepare_data(self, batch):
        x_data = []
        y_data = []
        for i in batch:
            x = self.__image_data_generator.random_transform(i)
            x_data.append(tf.math.reduce_mean(x, axis=2, keepdims=True))
            y_data.append(x if not self.__is_hsv else rgb_to_hsv(i))
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data, y_data

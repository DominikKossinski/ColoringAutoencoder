from typing import Tuple

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

    train_x, train_y = prepare_data(train, "Train dataset", nb_samples, size, is_hsv)

    val = tf_ds.load(name=name, split=tf_ds.Split.TRAIN)
    val = val.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)

    val_x, val_y = prepare_data(val, "Validation dataset", nb_samples // 5, size, is_hsv)
    return train_x, train_y, val_x, val_y


def prepare_data(dataset, name, n, size, is_hsv: bool):
    samples = list(dataset)[:n]
    data_x = []
    data_y = []
    for element in tqdm(samples, desc=name):
        img = Image.fromarray(np.array(element["image"]))
        if size is not None:
            img = img.resize(size)
        x = np.mean(np.asarray(img) / 255.0, axis=2, dtype="float32")
        y = np.asarray(img, dtype="float32") / 255.0
        if is_hsv:
            y = rgb_to_hsv(y)
        data_x.append(x)
        data_y.append(y)
    data_x = np.array(data_x, dtype='float32')
    data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], data_x.shape[2], 1))
    data_y = np.array(data_y, dtype='float32')
    return data_x, data_y

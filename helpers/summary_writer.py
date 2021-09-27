import os
import shutil
from datetime import datetime
from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow import summary
from tensorflow.image import hsv_to_rgb
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow_io.python.experimental.color_ops import lab_to_rgb

from autoencoders.AutoEncoderFormat import AutoEncoderFormat
from helpers.img_helper import load_image


class Scalars(Enum):
    loss = "loss"


class SummaryWriter:

    def __init__(self, logs_dir: str):
        self.__logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        if len(os.listdir(logs_dir)) > 0:
            shutil.rmtree(logs_dir)
        self.__train_writer = summary.create_file_writer(os.path.join(logs_dir, "train"))
        self.__val_writer = summary.create_file_writer(os.path.join(logs_dir, "val"))

    def write_val_loss(self, val_loss, step):
        with self.__val_writer.as_default():
            summary.scalar(Scalars.loss.value, val_loss, step=step)

    def write_train_loss(self, train_loss, step):
        with self.__train_writer.as_default():
            summary.scalar(Scalars.loss.value, train_loss, step=step)

    def write_val_image(self, name, val_image, step):
        with self.__val_writer.as_default():
            summary.image(name, np.array([val_image]), step)


class SummaryCallback(tf.keras.callbacks.Callback):

    def __init__(self, auto_encoder):
        super(SummaryCallback, self).__init__()
        self.__auto_encoder = auto_encoder
        self.__logs_dir = os.path.join(self.__auto_encoder.get_models_path(), "logs")
        self.__val_models_dir = os.path.join(self.__auto_encoder.get_models_path(), "val_models")
        os.makedirs(self.__val_models_dir, exist_ok=True)
        os.makedirs(self.__logs_dir, exist_ok=True)
        self.__writer = SummaryWriter(self.__logs_dir)

    def on_epoch_begin(self, epoch, logs=None):
        print(f'Starting {self.__auto_encoder.get_name()}: epoch {epoch} starts at {datetime.now().time()}')

    def on_epoch_end(self, epoch, logs=None):
        print(f'Evaluating {self.__auto_encoder.get_name()}: epoch {epoch} ends at {datetime.now().time()}')
        self.__writer.write_train_loss(logs["loss"], epoch)
        self.__writer.write_val_loss(logs["val_loss"], epoch)
        if epoch % 10 == 0:
            self.__auto_encoder.save(dir_path=self.__val_models_dir,
                                     name=f'{self.__auto_encoder.get_name()}_{epoch}')
            names = ['kwiat']
            for i in names:
                ae_format = self.__auto_encoder.get_format()
                (colored, black) = load_image(i + ".png", ae_format, size=(150, 150))
                y = self.__auto_encoder.predict(black)
                if ae_format == AutoEncoderFormat.HSV:
                    y = hsv_to_rgb(y)
                elif ae_format == AutoEncoderFormat.LAB:
                    y = y * np.array([100, 255, 255]) - np.array([0, 128, 128])
                    y = lab_to_rgb(y)
                else:
                    raise ValueError(f"No format: {ae_format}")
                self.__writer.write_val_image(i, y, epoch)

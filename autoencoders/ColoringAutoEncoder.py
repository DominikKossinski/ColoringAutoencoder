import os
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.constraints import max_norm

from helpers.img_predict_callback import ImgPredictCallback
from helpers.summary_writer import SummaryWriter


class AutoEncoderFormat(Enum):
    RGB = "RGB"
    HSV = "HSV"

    def __str__(self):
        return self.value


class ColoringAutoEncoder:

    def __init__(self, models_path: str, name: str, batch_size: int, format: AutoEncoderFormat):
        self.__format: AutoEncoderFormat = format
        self.__name: str = f'{name}{format}'
        self.__models_path: str = os.path.join(models_path, self.__name)
        os.makedirs(self.__models_path, exist_ok=True)
        self.__batch_size: int = batch_size
        self.__model = tf.keras.Sequential(name=name)
        self.__summary_writer: SummaryWriter = SummaryWriter()
        self.__add_encoder_layers()
        self.__add_decoder_layers()

    def __add_encoder_layers(self) -> None:
        self.__model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_constraint=max_norm(2.0),
                                       kernel_initializer='he_uniform'))

        self.__model.add(layers.BatchNormalization())
        self.__model.add(layers.ReLU())

        self.__model.add(
            layers.Conv2D(64, (3, 3), kernel_constraint=max_norm(2.0), kernel_initializer='he_uniform'))

        self.__model.add(layers.BatchNormalization())
        self.__model.add(layers.ReLU())

        self.__model.add(layers.Conv2D(128, (3, 3), kernel_initializer='he_uniform'))

        self.__model.add(layers.BatchNormalization())
        self.__model.add(layers.ReLU())

        self.__model.add(
            layers.Conv2D(128, (3, 3), kernel_initializer='he_uniform', kernel_constraint=max_norm(2.0),
                          padding='same'))

        self.__model.add(layers.BatchNormalization())
        self.__model.add(layers.ReLU())

    def __add_decoder_layers(self) -> None:
        self.__model.add(
            layers.Conv2DTranspose(128, (3, 3), kernel_constraint=max_norm(2.0), kernel_initializer='he_uniform'))

        self.__model.add(layers.BatchNormalization())
        self.__model.add(layers.ReLU())

        self.__model.add(
            layers.Conv2DTranspose(64, (3, 3), kernel_initializer='he_uniform', kernel_constraint=max_norm(2.0)))

        self.__model.add(layers.BatchNormalization())
        self.__model.add(layers.ReLU())

        self.__model.add(
            layers.Conv2DTranspose(32, (3, 3), kernel_initializer='he_uniform', kernel_constraint=max_norm(2.0)))

        self.__model.add(layers.BatchNormalization())
        self.__model.add(layers.ReLU())

        self.__model.add(
            layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', kernel_constraint=max_norm(2.0), padding='same',
                                   kernel_initializer='he_uniform'))

    def save(self, dir_path: str = None, name: str = None) -> None:
        if dir_path is None:
            dir_path = self.__models_path
        if name is None:
            name = self.__name
        self.__model.save(os.path.join(dir_path, f'{name}.h5'), save_format='h5')

    def build(self) -> None:
        self.__model.build((None, None, None, 1))
        self.__model.summary()

    def load(self, dir_path: str = None, name: str = None) -> None:
        if dir_path is None:
            dir_path = self.__models_path
        if name is None:
            name = self.__name
        self.__model.load_weights(os.path.join(dir_path, f'{name}.h5'))

    def compile(self, optimizer: tf.optimizers.Optimizer) -> None:
        self.__model.compile(optimizer=optimizer, loss='mse')

    def predict(self, x):
        return self.__model.predict(np.array([x]))[0]

    def get_name(self) -> str:
        return self.__name

    def get_models_path(self) -> str:
        return self.__models_path

    def train(self, x_train, y_train, x_val, y_val, epochs: int):
        checkpoint_path = os.path.join(self.__models_path, f'{self.__name}-checkpoint.h5')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True
        )
        img_predict_callback = ImgPredictCallback(self)
        history = self.__model.fit(
            x_train, y_train, self.__batch_size, epochs,
            validation_data=(x_val, y_val),
            callbacks=[save_callback, img_predict_callback]
        )
        # TODO show history

    def is_hsv(self):
        return self.__format == AutoEncoderFormat.HSV

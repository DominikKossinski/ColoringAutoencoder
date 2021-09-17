import os
from enum import Enum

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.constraints import max_norm

from helpers.git_ignore_helper import create_model_gitignore
from helpers.img_helper import AutoEncoderImageDataGenerator
from helpers.summary_writer import SummaryCallback


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
        create_model_gitignore(self.__models_path)
        self.__batch_size: int = batch_size
        self.__model = tf.keras.Sequential(name=name)
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

    def train(self, x_train, x_val, epochs: int):
        img_gen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=1.0,
            height_shift_range=1.0,
            horizontal_flip=True,
            vertical_flip=True
        )
        img_gen = AutoEncoderImageDataGenerator(self.is_hsv())
        checkpoint_path = os.path.join(self.__models_path, f'{self.__name}.h5')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            verbose=1,
            save_weights_only=True
        )
        summary_callback = SummaryCallback(self)
        print(f"Train len: {len(x_train)}")
        print(f"Steps: {len(x_train) // self.__batch_size}")
        # for e in range(epochs):
        #     batches = 0
        #     for x_batch in img_gen.flow(x_train):
        #         print(x_batch.shape)
        #         x_data = []
        #         y_data = []
        #         for i in x_batch:
        #             print(i.shape)
        #             x_data.append(tf.math.reduce_mean(i / 255.0, axis=2, keepdims=True))
        #             y_data.append(i if not self.is_hsv() else rgb_to_hsv(i))
        #         print(np.array(x_data).shape)
        #     for c in callbacks:
        #         c.on
        #         exit(-5)
        history = self.__model.fit(
            img_gen.flow(x_train, batch_size=self.__batch_size, subset="training"),
            batch_size=self.__batch_size,
            validation_data=img_gen.flow(x_val, batch_size=self.__batch_size, subset="validation"),
            callbacks=[save_callback, summary_callback],
            steps_per_epoch=len(x_train) // self.__batch_size, epochs=epochs
        )
        self.__show_and_save_history(history)

    def __show_and_save_history(self, history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title(f'{self.__name} loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(self.__models_path, f'{self.__name}_loss.png'))
        plt.show()
        loss_frame = pd.DataFrame(data={"loss": loss, "val_loss": val_loss})
        loss_frame.to_csv(os.path.join(self.__models_path, f'{self.__name}_loss.csv'), sep=";")

    def is_hsv(self):
        return self.__format == AutoEncoderFormat.HSV

    def get_name(self) -> str:
        return self.__name

    def get_models_path(self) -> str:
        return self.__models_path

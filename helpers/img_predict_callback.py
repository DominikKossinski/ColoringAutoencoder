import os
from datetime import datetime

import tensorflow as tf
from tensorflow.image import hsv_to_rgb
from tensorflow.keras.preprocessing import image as tf_image

from helpers.img_helper import load_image


class ImgPredictCallback(tf.keras.callbacks.Callback):

    def __init__(self, auto_encoder):
        super(ImgPredictCallback, self).__init__()
        self.__auto_encoder = auto_encoder
        self.__val_models_dir = os.path.join(self.__auto_encoder.get_models_path(), "val_models")
        self.__val_images_dir = os.path.join(self.__auto_encoder.get_models_path(), "val_images")
        os.makedirs(self.__val_models_dir, exist_ok=True)
        os.makedirs(self.__val_images_dir, exist_ok=True)

    def on_epoch_begin(self, epoch, logs=None):
        print(f'Starting {self.__auto_encoder.get_name()}: epoch {epoch} starts at {datetime.now().time()}')

    def on_epoch_end(self, epoch, logs=None):
        print(f'Evaluating {self.__auto_encoder.get_name()}: epoch {epoch} ends at {datetime.now().time()}')
        if epoch % 10 == 0:
            self.__auto_encoder.save(dir_path=self.__val_models_dir,
                                     name=f'{self.__auto_encoder.get_name()}_{epoch}')
            names = ['kwiat']
            for i in names:
                (colored, black) = load_image(i + ".png", size=(75, 75), is_hsv=self.__auto_encoder.is_hsv())
                y = self.__auto_encoder.predict(black)
                name = os.path.join(self.__val_images_dir, f"{i}_{epoch}.png")
                if self.__auto_encoder.is_hsv():
                    y = hsv_to_rgb(y)
                y *= 255.0
                img = tf_image.array_to_img(y, scale=False)
                img.save(name)

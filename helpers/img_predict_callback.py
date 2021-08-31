import os
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.image import hsv_to_rgb
from tensorflow.keras.preprocessing import image as tf_image

from helpers.img_helper import load_image


class ImgPredictCallback(tf.keras.callbacks.Callback):

    def __init__(self, auto_encoder):
        super(ImgPredictCallback, self).__init__()
        self.__auto_encoder = auto_encoder

    def on_epoch_begin(self, epoch, logs=None):
        print(f'Starting {self.__auto_encoder.get_name()}: epoch {epoch} starts at {datetime.now().time()}')

    def on_epoch_end(self, epoch, logs=None):
        print(f'Evaluating {self.__auto_encoder.get_name()}: epoch {epoch} ends at {datetime.now().time()}')
        if epoch % 10 == 0:
            self.__auto_encoder.save(name=f'{self.__auto_encoder.get_name()}_{epoch}.h5')
            names = ['kwiat']
            for i in names:
                (colored, black) = load_image(i + ".png", size=(75, 75), is_hsv=self.__auto_encoder.is_hsv())
                y = self.__auto_encoder.predict(black)
                name = os.path.join(self.__auto_encoder.get_models_path(), f"{i}_{epoch}.png")
                if self.__auto_encoder.is_hsv():
                    y = hsv_to_rgb(y)
                img = tf_image.array_to_img(y)
                img.save(name)

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as st
import tensorflow as tf
import tensorflow.keras.layers as lay
import tensorflow_datasets as tfds
from PIL import Image
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.preprocessing import image
from tqdm import tqdm


class Autoncoder:

    def add_encoder_layers(self):
        self.model.add(lay.Conv2D(32, (3, 3), activation='relu', kernel_constraint=max_norm(2.0),
                                  kernel_initializer='he_uniform'))

        self.model.add(lay.BatchNormalization())
        self.model.add(lay.ReLU())

        self.model.add(
            lay.Conv2D(64, (3, 3), kernel_constraint=max_norm(2.0), kernel_initializer='he_uniform'))

        self.model.add(lay.BatchNormalization())
        self.model.add(lay.ReLU())

        self.model.add(lay.Conv2D(128, (3, 3), kernel_initializer='he_uniform'))

        self.model.add(lay.BatchNormalization())
        self.model.add(lay.ReLU())

        self.model.add(
            lay.Conv2D(128, (3, 3), kernel_initializer='he_uniform', kernel_constraint=max_norm(2.0), padding='same'))

        self.model.add(lay.BatchNormalization())
        self.model.add(lay.ReLU())

    def add_decoder_layers(self):
        self.model.add(
            lay.Conv2DTranspose(128, (3, 3), kernel_constraint=max_norm(2.0), kernel_initializer='he_uniform'))

        self.model.add(lay.BatchNormalization())
        self.model.add(lay.ReLU())

        self.model.add(
            lay.Conv2DTranspose(64, (3, 3), kernel_initializer='he_uniform', kernel_constraint=max_norm(2.0)))

        self.model.add(lay.BatchNormalization())
        self.model.add(lay.ReLU())

        self.model.add(
            lay.Conv2DTranspose(32, (3, 3), kernel_initializer='he_uniform', kernel_constraint=max_norm(2.0)))

        self.model.add(lay.BatchNormalization())
        self.model.add(lay.ReLU())

        self.model.add(
            lay.Conv2DTranspose(3, (3, 3), activation='sigmoid', kernel_constraint=max_norm(2.0), padding='same',
                                kernel_initializer='he_uniform'))

    def save(self, path):
        self.model.save(path, save_format="h5")

    def show_and_save_history(self, history):
        print(history.history.keys())
        path = os.path.join("models", self.name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model {} loss'.format(self.name))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(path, self.name + '_loss.png'))
        plt.show()
        file = os.path.join("models", self.name, self.name + "_loss.csv")
        file = open(file, "w+")
        loss = list(loss)
        val_loss = list(val_loss)
        loss_string = "loss;" + ";".join(str(x) for x in loss) + "\n"
        val_loss_string = "val_loss;" + ";".join(str(x) for x in val_loss) + "\n"
        file.write(loss_string)
        file.write(val_loss_string)
        file.close()

    def build(self):
        self.model.build((None, None, None, 1))
        self.model.summary()

    def load(self, path):
        self.model.load_weights(path)

    def compile(self, optimizer):
        self.model.compile(optimizer=optimizer, loss='mse')

    def predict(self, x):
        return self.model.predict(np.array([x]))[0] * 255.0

    def train(self, x_train, y_train, x_val, y_val, epochs):
        path = os.path.join("models", self.name, self.name + ".h5")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=path,
            verbose=1,
            save_best_only=True
        )
        generate_callback = GenerateCallback(self.name, self.model)
        history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=self.batch_size,
                                 epochs=epochs,
                                 callbacks=[save_callback, generate_callback])
        self.show_and_save_history(history)

    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size
        self.model = tf.keras.Sequential()
        self.add_encoder_layers()
        self.add_decoder_layers()


class GenerateCallback(tf.keras.callbacks.Callback):

    def __init__(self, name, model):
        super().__init__()
        self.name = name
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        print('Starting {}: epoch {} ends at {}'.format(self.name, epoch, datetime.now().time()))

    def on_epoch_end(self, epoch, logs=None):
        print('Evaluating {}: epoch {} ends at {}'.format(self.name, epoch, datetime.now().time()))
        if epoch % 10 == 0:
            path = os.path.join("models", self.name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(os.path.join(path, self.name + "_{}.h5".format(epoch)), save_format="h5")
            names = ['kwiat']
            for i in names:
                (colored, black) = load_test_img(i + ".png", size=(75, 75))
                y = self.model.predict(np.array([black]))[0] * 255.0
                name = os.path.join(path, self.name + "_" + i + "_{}.png".format(epoch))
                img = image.array_to_img(y, scale=False)
                img.save(name)


def load_test_img(path, size=None):
    img = Image.open(path)
    if size is not None:
        img = img.resize(size)
    data = np.asanyarray(img, dtype='float32') / 255.0
    black = np.mean(data, axis=2)
    black = np.reshape(black, (black.shape[0], black.shape[1], 1))
    return data, black


def prepare_data(dataset, name, nb_samples, size):
    a = list(dataset)[:nb_samples]
    data_x = []
    data_y = []
    bar = tqdm(total=len(a), desc=name)
    for data in a:
        x = np.array(np.array(data["image"], dtype='float32') / 255.0, dtype='float32')
        if size is not None:
            x = st.resize(x, (size, size))
        x = np.array(x, dtype='float32')
        data_y.append(x)
        x = np.mean(x, axis=2)
        data_x.append(x)
        bar.update(1)
    bar.close()
    data_x = np.array(data_x, dtype='float32')
    data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], data_x.shape[2], 1))
    data_y = np.array(data_y, dtype='float32')
    return data_x, data_y


def load_data(name, nb_samples, size=100):
    train = tfds.load(name=name, split=tfds.Split.TEST)
    train = train.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)

    train_x, train_y = prepare_data(train, "Train dataset", nb_samples, size)

    val = tfds.load(name=name, split=tfds.Split.TRAIN)
    val = val.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)

    val_x, val_y = prepare_data(val, "Validation dataset", nb_samples // 5, size)
    return train_x, train_y, val_x, val_y


def setup_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == '__main__':
    setup_gpu()

    x_train, y_train, x_val, y_val = load_data('oxford_flowers102', 5000, 75)
    autoencoder = Autoencoder('Autoencoder', 32)
    opt = tf.keras.optimizers.Adam()
    autoencoder.build()
    autoencoder.compile(opt)
    autoencoder.train(x_train, y_train, x_val, y_val, 201)

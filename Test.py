from tensorflow.keras.preprocessing import image

from Autoencoder import load_test_img, Autoencoder, setup_gpu

if __name__ == '__main__':
    setup_gpu()
    (color, black) = load_test_img('kwiat.png', (960, 640))
    autoencoder = Autoencoder('Test', 64)
    autoencoder.build()
    autoencoder.load('models/Autoencoder/Autoencoder_0.h5')
    y = autoencoder.predict(black)
    img = image.array_to_img(y, scale=False)
    img.save('colored.png')
    black *= 255.0
    img = image.array_to_img(black, scale=False)
    img.save('black.png')
    color *= 255.0
    img = image.array_to_img(color, scale=False)
    img.save('original.png')
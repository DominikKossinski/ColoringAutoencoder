from argparse import ArgumentParser

import numpy as np
from tensorflow.image import hsv_to_rgb
from tensorflow.keras.preprocessing import image
from tensorflow_io.python.experimental.color_ops import lab_to_rgb

from autoencoders.ColoringAutoEncoder import AutoEncoderFormat, ColoringAutoEncoder
from helpers.img_helper import load_image
from helpers.tf_helper import setup_gpu


def setup_args_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--path", help="Models directory path", type=str, default="models")
    arg_parser.add_argument("--name", help="Model name", type=str, default="AutoEncoder")
    arg_parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=32)
    return arg_parser


def main(args):
    setup_gpu()
    (color, black) = load_image('kwiat.png', AutoEncoderFormat.RGB, (960, 640))

    rgb_auto_encoder = ColoringAutoEncoder(args.path, args.name, args.batch_size, AutoEncoderFormat.RGB, True)
    rgb_auto_encoder.build()
    rgb_auto_encoder.load()
    y_rgb = rgb_auto_encoder.predict(black)
    y_rgb *= 255.0
    img = image.array_to_img(y_rgb, scale=False)
    img.save('colored_rgb.png')

    hsv_auto_encoder = ColoringAutoEncoder(args.path, args.name, args.batch_size, AutoEncoderFormat.HSV, True)
    hsv_auto_encoder.build()
    hsv_auto_encoder.load()
    y_hsv = hsv_auto_encoder.predict(black)
    y_rgb = hsv_to_rgb(y_hsv)
    y_rgb *= 255.0
    img = image.array_to_img(y_rgb, scale=False)
    img.save('colored_hsv.png')

    lab_auto_encoder = ColoringAutoEncoder(args.path, args.name, args.batch_size, AutoEncoderFormat.LAB, True)
    lab_auto_encoder.build()
    lab_auto_encoder.load()
    img = lab_auto_encoder.predict(black)
    img = img * np.array([100, 255, 255]) - np.array([0, 128, 128])
    img = lab_to_rgb(img)
    img *= 255
    img = image.array_to_img(img, scale=False)
    img.save('colored_lab.png')

    black *= 255.0
    img = image.array_to_img(black, scale=False)
    img.save('black.png')
    color *= 255.0
    img = image.array_to_img(color, scale=False)
    img.save('original.png')


if __name__ == '__main__':
    parser = setup_args_parser()
    main(parser.parse_args())

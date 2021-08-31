from argparse import ArgumentParser

from tensorflow.image import hsv_to_rgb
from tensorflow.keras.preprocessing import image

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
    (color, black) = load_image('kwiat.png', (960, 640))

    rgb_auto_encoder = ColoringAutoEncoder(args.path, args.name, args.batch_size, AutoEncoderFormat.RGB)
    rgb_auto_encoder.build()
    rgb_auto_encoder.load()
    y_rgb = rgb_auto_encoder.predict(black)
    y_rgb *= 255.0
    img = image.array_to_img(y_rgb, scale=False)
    img.save('colored_rgb.png')

    hsv_auto_encoder = ColoringAutoEncoder(args.path, args.name, args.batch_size, AutoEncoderFormat.HSV)
    hsv_auto_encoder.build()
    hsv_auto_encoder.load()
    y_hsv = hsv_auto_encoder.predict(black)
    y_rgb = hsv_to_rgb(y_hsv)
    y_rgb *= 255.0
    img = image.array_to_img(y_rgb, scale=False)
    img.save('colored_hsv.png')

    black *= 255.0
    img = image.array_to_img(black, scale=False)
    img.save('black.png')
    color *= 255.0
    img = image.array_to_img(color, scale=False)
    img.save('original.png')


if __name__ == '__main__':
    parser = setup_args_parser()
    main(parser.parse_args())

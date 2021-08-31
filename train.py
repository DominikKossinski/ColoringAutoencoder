from argparse import ArgumentParser

import tensorflow as tf

from autoencoders.ColoringAutoEncoder import AutoEncoderFormat, ColoringAutoEncoder
from helpers.img_helper import load_data
from helpers.tf_helper import setup_gpu


def setup_args_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-e", "--epochs", help="Number of training epochs", type=int, default=201)
    arg_parser.add_argument("--path", help="Models directory path", type=str, default="models")
    arg_parser.add_argument("--name", help="Model name", type=str, default="AutoEncoder")
    arg_parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=32)
    arg_parser.add_argument(
        "-f", "--format", type=AutoEncoderFormat, choices=list(AutoEncoderFormat), default=AutoEncoderFormat.RGB
    )
    return arg_parser


def main(args) -> None:
    setup_gpu()
    x_train, y_train, x_val, y_val = load_data('oxford_flowers102', 5000, 75, args.format == AutoEncoderFormat.HSV)
    auto_encoder = ColoringAutoEncoder(args.path, args.name, args.batch_size, args.format)
    optimizer = tf.keras.optimizers.Adam()
    auto_encoder.build()
    auto_encoder.compile(optimizer)
    auto_encoder.train(x_train, y_train, x_val, y_val, args.epochs)


if __name__ == '__main__':
    parser = setup_args_parser()
    main(parser.parse_args())

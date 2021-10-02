# Coloring Autoencoder

An artificial neural network implemented with Tensorflow 2. Contains only conventional layers. Network trained on 6000
images (75 x 75 px) from
[Oxford Flowers 102](https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/).

## Training

To start model training type command:

```
python train.py 
    -h, --help                                  show this help message and exit
    -e EPOCHS, --epochs EPOCHS                  Number of training epochs
    --path PATH                                 Models directory path
    --name NAME                                 Model name
    -bs BATCH_SIZE, --batch-size BATCH_SIZE     Batch size
    -f {RGB,HSV,LAB}, --format {RGB,HSV,LAB}    Output image format
    -da, --data-ag                              Data agumentation
                
```

## Test

To test pretrained models type command:
```
python test.py
    -h, --help                                  show this help message and exit
    --path PATH                                 Models directory path
    --name NAME                                 Model name
    -bs BATCH_SIZE, --batch-size BATCH_SIZE     Batch size

```

During the training open next command line and type, to start Tensorboard:

```
tensorboard --logdir models/<Your model name><Image format>/logs
```

Then open url [localhost:6006/](localhost:6006/) to see generated summaries.

## Jupyter-notebook

The jupyter-notebook *examples.ipynb* contains code, that allows to test pretrained model on images given by user.

To start jupyter server type:

```commandline
jupyter-notebook examples.ipynb
```

## Images examples

### Input
| Original image | Input Image
| --- | --- |
| ![Input image](original.png) | ![Black and white image](black.png) |

### Output
| RGB | HSV| LAB |
| --- | --- |  --- |
| ![RGB](colored_rgb.png) | ![RGB](colored_hsv.png) | ![RGB](colored_lab.png)

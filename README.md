# Coloring Autoencoder

An artificial neural network implemented with Tensorflow 2. Contains only conventional layers. Network trained on 6000
images (75 x 75 px) from
[Oxford Flowers 102](https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/).

## Training

To start model training type command:

```

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

Example:

Input image:
![Input image](original.png)

Black and white image:
![Black and white image](black.png)

Colored image:
![Colored image](colored.png)

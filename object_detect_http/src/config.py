import torch
import os

# Pytorch install: https://pytorch.org/get-started/locally/

# TODO: put in paths to source and destination pic files
# TODO: put in path to model
# TODO: os environment variables load
# TODO: remove any config variables not needed, and assess if CUDA needs to now be CPU

SOURCE_CONNECTION = os.getenv('SOURCE_CONNECTION_STRING')
SOURCE_CONTAINER = os.getenv('SOURCE_CONTAINER')
MODEL_LOCATION = os.getenv('MODEL_LOCATION')
MODEL_NAME = os.getenv('MODEL_NAME')
COMPUTATION_DEVICE = 'cpu' # using cpu for azure function app for now instead of 'cuda'
DEVICE = torch.device(COMPUTATION_DEVICE)

CLASSES = 3

# computational setup

BATCH_SIZE = 9      # increase / decrease according to GPU memory - 9 works for GeForce 1070, 8GB RAM
RESIZE_TO = 640     # resize the image for training and transforms - TODO: how big can we go?  is bigger better?
NUM_EPOCHS = 100    # number of epochs to train for
# TODO: how do epochs relate to training quality (output graphs) - other parameters?

LEARNING_RATE_START = 0.001     # default 0.001
LEARNING_RATE_DECAY = 0.95      # starting point for exponential decay of learning rate
MOMENTUM_START = 0.98           # default 0.9
WEIGHT_DECAY = 0.00025           # default 0.0005

# classes: 0 index is reserved for background
NUM_CLASSES = 3
CLASSES = ['background', 'valve', 'guage']


# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../model_outputs/'
SAVE_PLOTS_EPOCH = 25  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 25   # save model after these many epochs

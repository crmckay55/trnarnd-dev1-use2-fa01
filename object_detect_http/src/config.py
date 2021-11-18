# config.py
# Chris McKay
# v1.0 2021-11-14
# Loads relevant envrionment variables from function app environment

import os


SOURCE_CONNECTION = os.getenv('SOURCE_CONNECTION_STRING')
SOURCE_CONTAINER = os.getenv('SOURCE_CONTAINER')
MODEL_LOCATION = os.getenv('MODEL_LOCATION')
MODEL_NAME = os.getenv('MODEL_NAME')
COMPUTATION_DEVICE = 'cpu' # using cpu for azure function app for now instead of 'cuda'

CLASSES = int(os.getenv('MODEL_CLASSES'))
import struct
import importlib.resources
import os

from tensorflow import keras  # type:ignore

SUPPORTED_DATASETS = ["CIFAR10", "CIFAR100", "GTSRB", "IMAGENET"]

SUPPORTED_MODELS = [
    "ResNet18",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "ResNet56",
    "ResNet110",
    "ResNet1202",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "GoogLeNet",
    "MobileNetV2",
    "InceptionV3",
    "Vgg11_bn",
    "Vgg13_bn",
    "Vgg16_bn",
    "Vgg19_bn",
]

MODULE_PATH = str(importlib.resources.files(__package__))
DEFAULT_DATASET_PATH = os.path.join(MODULE_PATH, "../datasets")
DEFAULT_MODEL_PATH = os.path.join(MODULE_PATH, "../models")

INJECTED_LAYERS_TYPES = (keras.layers.Conv2D, keras.layers.Dense)


def float32_to_int(value):
    bytes_value = struct.pack("f", value)
    int_value = struct.unpack("I", bytes_value)[0]
    return int_value


def int_to_float32(int_value):
    bytes_value = struct.pack("I", int_value)
    return struct.unpack("f", bytes_value)[0]

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

REPORT_HEADER = (
    "inj_id",
    "layer,weigths",
    "bit_pos",
    "n_injections",
    "top_1_correct",
    "top_5_correct",
    "top_1_robust",
    "top_5_robust",
    "masked",
    "non_critical",
    "critical",
)

DEFAULT_REPORT_DIR = os.path.join(MODULE_PATH, "../reports")

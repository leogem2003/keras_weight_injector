import os
from benchmark_models.tf_utils import create_manipulated_model

from benchmark_models.utils import (
    SUPPORTED_MODELS,
    SUPPORTED_DATASETS,
    get_device,
    get_loader,
    load_network,
)
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow import keras
import argparse

import numpy as np

import nobuco
from nobuco import ChannelOrder, converter, ChannelOrderingStrategy

from benchmark_models.inference_tools.pytorch_inference_manager import (
    PTInferenceManager,
)
from benchmark_models.inference_tools.tf_inference_manager import TFInferenceManager


@converter(
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.functional.max_pool2d,
    channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS,
)
def convert_max_pool2d(self: torch.nn.modules.pooling.MaxPool2d, x):

    if not self.dilation in [1, (1,1)]:
        raise NotImplementedError(f'Conversion with dilation ({self.dilation}) different than (1,1) is not supported')

    def func(x):
        if isinstance(self.padding, tuple):
            pad_h, pad_w = self.padding
        elif isinstance(self.padding, int):
            pad_h = self.padding
            pad_w = self.padding
        else:
            raise ValueError(f'Padding format ({self.padding}) not supported')
        x_padded = tf.pad(x, [[0,0], [pad_h, pad_h], [pad_w, pad_w], [0,0]], constant_values=-np.infty)

        pooling =  keras.layers.MaxPooling2D(
            self.kernel_size,
            strides=self.stride,
            padding="valid",
            data_format="channels_last",
        )

        x = pooling(x_padded)

        return x


    return func

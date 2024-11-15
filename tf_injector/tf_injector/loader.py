from tf_injector.utils import DEFAULT_MODEL_PATH, DEFAULT_DATASET_PATH
from tf_injector.preprocessing import preprocessors

import tensorflow as tf  # type:ignore
from tensorflow import keras  # type:ignore
import os

loaders = {  # (test, test_labels)
    "CIFAR10": lambda: tf.keras.datasets.cifar10.load_data()[1],
    "CIFAR100": lambda: tf.keras.datasets.cifar100.load_data()[1],
}

import numpy as np
import tqdm


def test_dt_distance(d1, d2):
    for dt1_l, dt2_l in tqdm.tqdm(zip(d1, d2), "running dataset test"):
        dt1 = dt1_l[0]
        dt2 = dt2_l[0]
        diff = dt1 - dt2
        assert not diff.any(), f"{diff[diff!=0]}"


def load_network(
    network_name: str,
    dataset_name: str,
    model_path=DEFAULT_MODEL_PATH,
    dataset_path=DEFAULT_DATASET_PATH,
) -> tuple[keras.Model, tf.data.Dataset]:
    model_path = os.path.join(model_path, dataset_name, network_name + ".keras")
    model = keras.models.load_model(model_path)

    dataset_path = os.path.join(dataset_path, dataset_name)
    other_dataset = tf.data.Dataset.load(dataset_path, compression="GZIP")

    dataset = tf.data.Dataset.from_tensor_slices(loaders[dataset_name]()).map(
        preprocessors[dataset_name]
    )
    # test_dt_distance(
    #    dataset.as_numpy_iterator(),
    #    other_dataset.as_numpy_iterator(),
    # )
    return model, dataset

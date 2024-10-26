from tf_injector.utils import DEFAULT_MODEL_PATH, DEFAULT_DATASET_PATH

import tensorflow as tf  # type:ignore
from tensorflow import keras  # type:ignore
import os


def load_network(
    network_name: str,
    dataset_name: str,
    model_path=DEFAULT_MODEL_PATH,
    dataset_path=DEFAULT_DATASET_PATH,
) -> tuple[keras.Model, tf.data.Dataset]:
    model_path = os.path.join(model_path, dataset_name, network_name + ".keras")
    model = keras.models.load_model(model_path)

    dataset_path = os.path.join(dataset_path, dataset_name)
    dataset = tf.data.Dataset.load(dataset_path)

    return model, dataset

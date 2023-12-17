import os

import tensorflow as tf
from tensorflow import keras

def load_converted_tf_network(network_name: str) -> keras.Model:
    """
    Load the converted network from 
    """

    # Load the weights
    network_path = os.path.join('models', 'converted-tf', f'{network_name}.keras')

    return keras.models.load_model(network_path)


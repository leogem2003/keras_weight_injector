import os
import tensorflow as tf
import keras
from keras.src.engine import functional
from keras.src.layers import Layer, InputLayer
from keras.src.engine.keras_tensor import KerasTensor

from collections import defaultdict
from typing import Callable, Iterable, List, Dict, Union, Optional
from collections.abc import Sized


def load_converted_tf_network(network_name: str) -> keras.Model:
    """
    Load the converted keras network from .keras local file.

    """

    # Load the weights
    network_path = os.path.join("models", "converted-tf", f"{network_name}.keras")

    return keras.models.load_model(network_path)


def deep_clone_function_factory(
    inner_clone_function: Callable[[Layer, Layer], Optional[Layer]],
    verbose=False,
    copy_weights=True,
) -> Callable[[Layer], Layer]:
    def _clone_function(layer):
        if verbose:
            print(f"Cloning Layer Name: {layer.name} Type:{type(layer)}")

        if isinstance(layer, keras.Model):
            if verbose:
                print(f"Layer {layer.name} is a sub-Model. Cloning it recursively")
            cloned_submodel = keras.models.clone_model(
                layer, clone_function=_clone_function
            )
            if copy_weights:
                cloned_submodel.set_weights(layer.get_weights())
            return cloned_submodel
        cloned_layer = layer.__class__.from_config(layer.get_config())
        maybe_changed_layer = inner_clone_function(cloned_layer, layer)
        if maybe_changed_layer is layer:
            raise ValueError('Clone function returned the old layer. Never return the old_layer from the clone function.')
        if maybe_changed_layer is not None:
            return maybe_changed_layer
        else:
            return cloned_layer

    return _clone_function


def create_manipulated_model(
    model: keras.Model,
    clone_function: Callable[[Layer, Layer], Optional[Layer]] = lambda cloned_layer, old_layer: None
    verbose=False,
    copy_weights=True,
) -> keras.Model:
    """
    Clones and manipulate a Keras functional model, applying the clone_function also to nested submodels.

    This function is a wrapper for ```keras.models.clone_model()```

    ## Args:
        - ```model```: A Keras Functional model that will be cloned. This model will not be changed in the process.

        - ```clone_function```: An optional function that can make changes in the cloned model, with respect to the original function.
            If not specified the model will be cloned as it is without modifications. 
            The function takes in input two keras.Layer (```cloned_layer, old_layer```) and returns a ```keras.Layer``` or ```None```.
            The first argument contains the cloned layer that would be placed in the new model unless another layer is returned in this function.
            The second argument contains the layer to be cloned from the old graph. That can be used to get information about the old model that is going
            to be cloned.
            If the function returns ```None``` or exactly ```cloned_layer``` then ```new_layer``` in the source model will be inserted as it is in the new cloned model
            If the function returns another layer, then that layer replaces the ```new_layer``` in the cloned model with a custom layer.
            In no circumstanes ```old_layer``` should be returned from this function, otherwise a ```ValueError``` will be thrown.

        - ```verbose```: Print to stdout information about the layers when they are cloned. Defaults to ```False```.

        - ```copy_weights```: Copy the weights of the old model in the cloned model. This may throw errors if ```clone_function``` inserts layers with weights. Defaults to ```True```.
            
    
    ## Usage Example:
    ```
    import keras
    from tf_utils import clone_function

    class FaultInjector(keras.Layer):
        pass

    # Example of a function for adding a fault injection layer after a layer named "conv_2d_1" 
    def clone_function(cloned_layer, old_layer):
        if old_layer.name == "conv_2d_1":
            return keras.Sequentual([cloned_layer, FaultInjector()])
        else:
            return None #or cloned_layer

    model_to_clone = ...        
    
    cloned_model = create_manipulated_model(model_to_clone, clone_function)
    ```
    """
    clone_fn = deep_clone_function_factory(
        clone_function, verbose=verbose, copy_weights=copy_weights
    )
    cloned_model = keras.models.clone_model(model, clone_function=clone_fn)
    if copy_weights:
        cloned_model.set_weights(model.get_weights())
    return cloned_model

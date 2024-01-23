import os
from tf_utils import create_manipulated_model

from models.utils import load_ImageNet_validation_set

from models.utils import load_CIFAR10_datasets

from utils import SUPPORTED_MODELS_LIST, get_device, get_loader, load_network
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow import keras
import argparse

import nobuco
from nobuco import ChannelOrder, converter, ChannelOrderingStrategy

from InferenceManager import InferenceManager
from TFInferenceManager import TFInferenceManager

# NOBUCO converters for missing operators


class PrintShapeLayer(keras.layers.Layer):
    def __init__(self, prev_name="unknown", **kwargs):
        super(PrintShapeLayer, self).__init__(**kwargs)
        self.prev_name = prev_name

    def call(self, inputs):
        if hasattr(inputs, "shape"):
            # Print the desired message with the input shape
            print(f"Hello I'm {self.prev_name}, my shape is {inputs.shape}")
        else:
            # Print the desired message with the input shape
            print(f"Hello I'm {self.prev_name}, I have no shape")
        return inputs


@converter(
    torch.nn.Sequential,
    channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER,
)
def convert_Sequential(self, x):
    kwargs = {"return_outputs_pt": True, "trace_shape": True}
    # tf_sequential = tf.keras.Sequential()
    temp_out = x

    # for i, module in enumerate(self.children()):
    #    b, c, h, w = temp_out.shape
    #    seq_layer, seq_out = nobuco.pytorch_to_keras(module, input_shapes={input: (None, c, h, w)}, args=(temp_out,), **kwargs)
    #    temp_out = seq_out
    #    tf_sequential.add(seq_layer)

    tf_layer_list = []

    for i, module in enumerate(self.children()):
        seq_layer, seq_out = nobuco.pytorch_to_keras(
            module,
            input_shapes={input: tuple([None] + list(temp_out.shape[1:]))},
            args=(temp_out,),
            inputs_channel_order=ChannelOrder.TENSORFLOW,
            outputs_channel_order=ChannelOrder.TENSORFLOW,
            **kwargs,
        )
        temp_out = seq_out
        tf_layer_list.append(seq_layer)

    def func(inp):
        temp = inp
        for layer in tf_layer_list:
            temp = layer(temp)
        return temp

    return func


@converter(
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.functional.max_pool2d,
    channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS,
)
def convert_Sequential(self: torch.nn.modules.pooling.MaxPool2d, x):
    def func(x):
        return keras.layers.MaxPooling2D(
            self.kernel_size,
            strides=self.stride,
            padding="same",
            data_format="channels_last",
        )(x)

    return func


def convert_pt_to_tf(network: Module, loader: DataLoader):
    sample_image, _ = next(iter(loader))

    print("Starting conversion from PyTorch to Keras")
    print(f"Using dataset sample image of shape: {sample_image.shape}")

    b, h, w, c = sample_image.shape
    with torch.no_grad():
        keras_model = nobuco.pytorch_to_keras(
            network,
            args=(sample_image,),
            kwargs=None,
            input_shapes={input: (None, h, w, c)},
            trace_shape=True,
            inputs_channel_order=ChannelOrder.TENSORFLOW,
            outputs_channel_order=ChannelOrder.TENSORFLOW,
        )
        print("Conversion from PyTorch to Keras complete.")
    return keras_model


def parse_args():
    """
    Parse the argument of the network
    :return: The parsed argument of the network
    """

    parser = argparse.ArgumentParser(
        description="Run Inferences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--forbid-cuda",
        action="store_true",
        help="Completely disable the usage of CUDA. This command overrides any other gpu options.",
    )
    parser.add_argument(
        "--use-cuda", action="store_true", help="Use the gpu if available."
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Test set batch size"
    )
    parser.add_argument(
        "--network-name",
        "-n",
        type=str,
        required=True,
        help="Target network",
        choices=SUPPORTED_MODELS_LIST,
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Allow overwriting the existing network.",
    )

    parser.add_argument(
        "--output-path",
        "-p",
        type=str,
        help="Override output .keras file path. Default is models/converted-tf/{model_name}.keras.",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Entierly skip compared validation between PyTorch and converted TensorFlow model.",
    )

    parser.add_argument(
        "--skip-pt-validation",
        action="store_true",
        help="Validate only the Keras model, skipping the PyTorch validation.",
    )

    parser.add_argument(
        "--skip-manipulation",
        action="store_true",
        help="Skip Manipulation test",
    )

    parsed_args = parser.parse_args()

    return parsed_args


def fake_layer_factory(layer):
    print(f"Layer Name: {layer.name} Type:{type(layer)}")
    if isinstance(layer, keras.Model):
        cloned_submodel = keras.models.clone_model(
            layer, clone_function=fake_layer_factory
        )
        cloned_submodel.set_weights(layer.get_weights())
        return cloned_submodel
    if isinstance(layer, keras.layers.ReLU):
        return keras.Sequential([keras.layers.ReLU(), PrintShapeLayer(layer.name)])

    cloned_layer = layer.__class__.from_config(layer.get_config())
    return cloned_layer


def deep_clone_function_factory(inner_clone_function, verbose=False, copy_weights=True):
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
        maybe_cloned_layer = inner_clone_function(layer)
        if maybe_cloned_layer is not None:
            return maybe_cloned_layer

        cloned_layer = layer.__class__.from_config(layer.get_config())
        return cloned_layer

    return _clone_function


def prepare_loader(network_name, batch_size, permute_tf=False):
    # Load the dataset
    if "ResNet" in network_name:
        _, _, loader = load_CIFAR10_datasets(
            test_batch_size=batch_size, permute_tf=permute_tf
        )
        print(f"Using dataset: CIFAR10")
    else:
        loader = load_ImageNet_validation_set(
            batch_size=batch_size, image_per_class=1, permute_tf=permute_tf
        )
    return loader


def main(args):
    print("Running nobuco converter")
    print(f"Converting network {args.network_name} to PyTorch")

    output_path = args.output_path or os.path.join(
        "models", "converted-tf", f"{args.network_name}.keras"
    )

    if args.output_path is None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    skip_conversion = os.path.exists(output_path) and not args.overwrite

    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=True)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda, use_cuda=args.use_cuda)

    print(f"Using device {device}")

    # Load the network
    network = load_network(network_name=args.network_name, device=device)
    network.eval()

    _, conversion_loader = get_loader(
        network_name=args.network_name, batch_size=args.batch_size, permute_tf=False
    )

    if not skip_conversion:
        print(
            "STEP 1. [STARTING] Conversion from PyTorch to Keras using nobuco convertor"
        )
        with torch.no_grad():
            keras_model = convert_pt_to_tf(network, conversion_loader)
            print(
                "STEP 1. [COMPLETED] Conversion from PyTorch to Keras using nobuco convertor"
            )
        print("STEP 2. [STARTING] Saving converted model to .keras")
        keras_model.save(output_path)
        print("STEP 2. [COMPLETED] Saving converted model to .keras")
        print(
            f"Converted model saved to {output_path}. It can be loaded using keras.models.load_model('path/to/model')"
        )
    else:
        print(
            "STEP 1. [SKIPPED] Conversion from PyTorch to Keras using nobuco convertor. Model already present, to avoid skipping use --overwrite flag."
        )
        print(
            "STEP 2. [SKIPPED] Saving converted model to .keras. Model already present, to avoid skipping use --overwrite flag. "
        )

    print(f"STEP 3. [STARTED] Reloading Keras model from {output_path}.")
    reloaded_keras_model = keras.models.load_model(output_path)

    reloaded_keras_model.summary(expand_nested=True)

    _, validation_loader_tf = get_loader(
        network_name=args.network_name, batch_size=args.batch_size, permute_tf=True
    )

    _, validation_loader_pt = get_loader(
        network_name=args.network_name, batch_size=args.batch_size, permute_tf=False
    )

    print(f"STEP 3. [COMPLETED] Reloading Keras model from {output_path}.")
    if not args.skip_validation:
        if not args.skip_pt_validation:
            print("STEP 4. [STARTING] Validating PyTorch Model.")
            inference_executor = InferenceManager(
                network=network,
                network_name=args.network_name,
                device=device,
                loader=validation_loader_pt,
            )
            inference_executor.run_clean(save_outputs=False)
            print("STEP 4. [COMPLETED] Validating PyTorch Model.")
        else:
            print("STEP 4. [SKIPPED] Validating PyTorch Model.")
        # Run inference for the TF converted network to compare the value
        print("STEP 5. [STARTING] Validating converted Keras Model.")

        tf_inference_executor = TFInferenceManager(
            network=reloaded_keras_model,
            network_name=args.network_name,
            loader=validation_loader_tf,
        )
        tf_inference_executor.run_clean(save_outputs=False)
        print("STEP 5. [COMPLETED] Validating converted Keras Model.")
    else:
        print("STEP 4. [SKIPPED] Validating PyTorch Model.")
        print("STEP 5. [SKIPPED] Validating converted Keras Model.")

    if not args.skip_manipulation:
        print("STEP 6. [STARTING] Keras Model cloning and manipulation test.")

        def fake_injection_clone_function(layer, old_layer):
            if isinstance(
                layer,
                (
                    keras.layers.Conv2D,
                    keras.layers.BatchNormalization,
                    keras.layers.Dense,
                    keras.layers.Add,
                    keras.layers.Concatenate,
                    keras.layers.MaxPool2D,
                ),
            ):
                return keras.Sequential([layer, PrintShapeLayer(layer.name)])
            else:
                return None

        cloned_model = create_manipulated_model(
            reloaded_keras_model, fake_injection_clone_function, copy_weights=True
        )

        cloned_model.summary(expand_nested=True)
        tf_cloned_inference_executor = TFInferenceManager(
            network=cloned_model,
            network_name=args.network_name,
            loader=validation_loader_tf,
        )
        tf_cloned_inference_executor.run_clean()
        print("STEP 6. [COMPLETED] Keras Model cloning and manipulation test.")
    else:
        print("STEP 6. [SKIPPED] Keras Model cloning and manipulation test.")

    print("All done.")


if __name__ == "__main__":
    main(parse_args())

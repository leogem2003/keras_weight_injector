import os

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

# NOBUCO converters for missing operators


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
    channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS,
)
def convert_Sequential(self : torch.nn.modules.pooling.MaxPool2d, x):
    def func(x):
        return keras.layers.MaxPooling2D(self.kernel_size, strides=self.stride, padding="same", data_format="channels_last")(x)
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

    parsed_args = parser.parse_args()

    return parsed_args


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
    print(f"Converting network {args.network_name} to pythorch")

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

    _, conversion_loader = get_loader(network_name=args.network_name,
                        batch_size=args.batch_size, permute_tf=False)

    if not skip_conversion:
        print("Starting conversion to TensorFlow")
        with torch.no_grad():
            keras_model = convert_pt_to_tf(network, conversion_loader)
            print("Tensorflow conversion complete. See nobuco output for info.")

        keras_model.save(output_path)
        print(
            f'Converted model saved to {output_path}. It can be loaded using keras.models.load_model("path/to/model")'
        )
    else:
        print('Skipping conversion, reloading existing model.')

    reloaded_keras_model = keras.models.load_model(output_path)

    reloaded_keras_model.summary(expand_nested=True)

    if not args.skip_validation:
        from InferenceManager import InferenceManager
        from TFInferenceManager import TFInferenceManager

        print("Validation of the converted model")
        print("STEP 1. Running original model in PyTorch")

        _, validation_loader_pt = get_loader(network_name=args.network_name,
                        batch_size=args.batch_size, permute_tf=False)
        _, validation_loader_tf = get_loader(network_name=args.network_name,
                        batch_size=args.batch_size, permute_tf=True)

        inference_executor = InferenceManager(
            network=network,
            network_name=args.network_name,
            device=device,
            loader=validation_loader_pt,
        )
        inference_executor.run_clean(save_outputs=False)
        # Run inference for the TF converted network to compare the value
        print("STEP 2. Running converted model in TensorFlow")

        tf_inference_executor = TFInferenceManager(
            network=reloaded_keras_model,
            network_name=args.network_name,
            loader=validation_loader_tf,
        )
        tf_inference_executor.run_clean(save_outputs=False)
        print("Validation completed")
    else:
        print("Validation skipped")

    print("Done")


if __name__ == "__main__":
    main(parse_args())

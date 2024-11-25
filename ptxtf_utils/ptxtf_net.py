import importlib.util
import sys
import argparse

from tensorflow import keras
from torch import nn


def get_pt(path: str, name: str):
    spec = importlib.util.spec_from_file_location("pt_network", path)
    if spec is None:
        raise FileNotFoundError(f"Cannot find file {path}")

    pt_network = importlib.util.module_from_spec(spec)
    sys.modules["pt_network"] = pt_network

    if spec.loader is None:
        raise ValueError("Cannot load module")

    spec.loader.exec_module(pt_network)
    return getattr(pt_network, name)


def get_tf(path: str):
    return keras.models.load_model(path)


def get_pt_modules(pt_network: nn.Module, layers: tuple[nn.Module, ...]):
    modules = []
    for name, module in pt_network().named_modules():
        if isinstance(module, layers):  # type:ignore
            modules.append(name)
    return modules


def get_tf_modules(tf_network, layers):
    modules = []
    for layer in tf_network._flatten_layers(include_self=False, recursive=True):
        if isinstance(layer, layers):
            modules.append(layer.name)
    return modules


tf_layers = (keras.layers.Conv2D, keras.layers.Dense)
pt_layers = (nn.Conv2d, nn.Linear)


def main(args):
    pt_network = get_pt(*args.pt_path)
    tf_network = get_tf(args.tf_path)

    pt_names = get_pt_modules(pt_network, pt_layers)
    tf_names = get_tf_modules(tf_network, tf_layers)
    assert len(pt_names) == len(
        tf_names
    ), f"Cannot match netowrks layers: expected the same length, got {len(pt_names)}(PT) vs {len(tf_names)}(TF)"

    with open(args.output, "w") as f:
        f.write("PT,TF\n")
        f.writelines((f"{ptn},{tfn}\n" for ptn, tfn in zip(pt_names, tf_names)))


def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "pt_path",
        nargs=2,
        help="path to a .py file containing a PT network definition and the name of the network",
    )
    argparser.add_argument(
        "tf_path", help="path to a .keras file containing a keras network"
    )
    argparser.add_argument(
        "--output",
        "-o",
        help="save the output of the matching here",
        default="./out.txt",
    )
    return argparser.parse_args()


if __name__ == "__main__":
    main(parser())

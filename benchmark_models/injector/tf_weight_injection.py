from dataclasses import dataclass
import csv
import sys
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from benchmark_models.inference_tools.metric_evaluators import TopKAccuracy
from benchmark_models.inference_tools.tf_inference_manager import TFInferenceManager
from benchmark_models.utils import SUPPORTED_MODELS, SUPPORTED_DATASETS, get_loader
from benchmark_models.tf_utils import load_converted_tf_network
import argparse
from contextlib import contextmanager
from operator import attrgetter
from tabulate import tabulate
import struct
from tqdm import tqdm
import numpy as np
import csv
import os
from datetime import datetime

DEFAULT_REPORT_FOLDER = "reports"
INJECTED_LAYERS_TYPES_KERAS = (keras.layers.Conv2D, keras.layers.Dense)


def convert_weights_coords_from_pt_to_tf(weight_coords):
    if len(weight_coords) == 2:
        out_features, in_features = weight_coords
        return (in_features, out_features)
    if len(weight_coords) == 4:
        out_channels, in_channels, filter_height, filter_width = weight_coords
        return (filter_height, filter_width, in_channels, out_channels)
    raise NotImplementedError(
        f"The rearrangement when weights have {len(weight_coords)} dimensions is not handled"
    )


def load_fault_list(fault_list_path: str, is_channel_last=False, to_channel_last=True):
    layer_list = []
    injections = []
    with open(fault_list_path) as f:
        cr = csv.reader(f)
        next(cr, None)  # Skip Header
        for row in cr:
            try:
                inj_id, layer_name, weight_pos, bit = row
                inj_id = int(inj_id)
                if layer_name not in layer_list:
                    layer_list.append(layer_name)
                weight_pos = [
                    int(coord.strip()) for coord in weight_pos.strip("()").split(",")
                ]
                weight_pos_new = convert_weights_coords_from_pt_to_tf(weight_pos)
                bit = int(bit)
                injections.append((inj_id, layer_name, weight_pos_new, bit))
            except Exception as e:
                print(f"Error happened processing the following row: {row}")
                raise e from None
        return layer_list, injections


@contextmanager
def weight_bit_flip_applied(
    keras_model: keras.Model,
    layer: keras.layers.Layer,
    weight_coord: Tuple[int],
    bit: int,
):
    weights = layer.get_weights()
    selected_weight = weights[0][weight_coord]
    selected_weight_int = float32_to_int(selected_weight)
    selected_weight_int ^= 1 << bit
    new_weight = int_to_float32(selected_weight_int)
    weights[0][weight_coord] = new_weight
    layer.set_weights(weights)
    try:
        yield keras_model
    finally:
        weights[0][weight_coord] = selected_weight
        layer.set_weights(weights)


def float32_to_int(value):
    bytes_value = struct.pack("f", value)
    int_value = struct.unpack("I", bytes_value)[0]
    return int_value


def int_to_float32(int_value):
    bytes_value = struct.pack("I", int_value)
    return struct.unpack("f", bytes_value)[0]


def main(args):
    _, loader = get_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        permute_tf=True,
        dataset_path="../datasets",
    )

    tf_network = load_converted_tf_network(
        args.network_name,
        args.dataset,
        models_path="benchmark_models/models/converted-tf",
    )
    faulty_network = keras.models.clone_model(tf_network)
    faulty_network.set_weights(tf_network.get_weights())

    target_layers_list, injections = load_fault_list(
        args.fault_list, is_channel_last=False, to_channel_last=True
    )

    # Filter layers
    keras_conv_layers = [
        layer
        for layer in faulty_network._flatten_layers(include_self=False, recursive=True)
        if isinstance(layer, INJECTED_LAYERS_TYPES_KERAS)
    ]

    keras_conv_layers_names = list(map(attrgetter("name"), keras_conv_layers))

    print(
        tabulate(
            zip(target_layers_list, keras_conv_layers_names),
            headers=["PyTorch", "Keras"],
        )
    )

    print(f"Number of target layers: {len(target_layers_list)}")
    print(f"Number of conv layers in model: {len(keras_conv_layers)}")

    if len(target_layers_list) != len(keras_conv_layers):
        raise ValueError("Mismatching layer mapping")

    top_1_accuracy = TopKAccuracy(k=1)
    top_5_accuracy = TopKAccuracy(k=5)

    target_layer_mapping = dict(zip(target_layers_list, keras_conv_layers))
    inf_manager = TFInferenceManager(tf_network, "ResNet20", loader)
    inf_manager.run_clean()
    top_1_gold, top_1_acc = inf_manager.evaluate_metric(
        top_1_accuracy, use_faulty_outputs=False
    )
    top_5_gold, top_5_acc = inf_manager.evaluate_metric(
        top_5_accuracy, use_faulty_outputs=False
    )
    print(f"Accuracy TOP 1: {top_1_gold} {top_1_acc * 100}%")
    print(f"Accuracy TOP 5: {top_5_gold} {top_5_acc * 100}%")
    golden_labels = np.argmax(inf_manager.clean_output_scores, axis=-1)

    csv_headers = [
        "inj_id",
        "layer",
        "weigths",
        "bit_pos",
        "n_injections",
        "top_1_correct",
        "top_5_correct",
        "top_1_robust",
        "top_5_robust",
        "masked",
        "non_critical",
        "critical",
    ]

    gold_row = [
        "GOLDEN",
        None,
        None,
        None,
        inf_manager.clean_inference_counts,
        top_1_gold,
        top_5_gold,
        None,
        None,
        None,
        None,
    ]

    if args.output_path is None:
        report_folder = os.path.join(
            DEFAULT_REPORT_FOLDER, args.dataset, args.network_name
        )
        os.makedirs(report_folder, exist_ok=True)
        report_file_path = os.path.join(
            report_folder,
            f"{args.dataset}_{args.network_name}_{datetime.now().strftime('%y%m%d_%H%M')}.csv",
        )
    else:
        report_file_path = args.output_path

    write_header = not os.path.exists(report_file_path)
    with open(report_file_path, "a") as f:
        report_writer = csv.writer(f)
        if write_header:
            report_writer.writerow(csv_headers)
            report_writer.writerow(gold_row)

        for inj_num, injection in enumerate(
            tqdm(injections[args.resume_from :], leave=False)
        ):
            inj_id, target_layer_name, weight_coords, bit_pos = injection
            target_layer = target_layer_mapping[target_layer_name]
            with weight_bit_flip_applied(
                faulty_network, target_layer, weight_coords, bit_pos
            ):
                inf_manager.run_faulty(faulty_network)
                inf_count = inf_manager.faulty_inference_counts

                top_1_count, _ = inf_manager.evaluate_metric(
                    top_1_accuracy, use_faulty_outputs=True
                )
                top_5_count, _ = inf_manager.evaluate_metric(
                    top_5_accuracy, use_faulty_outputs=True
                )

                clean_out_scores = np.array(inf_manager.clean_output_scores)
                faulty_out_scores = np.array(inf_manager.faulty_output_scores)

                masked_count = (clean_out_scores == faulty_out_scores).all(axis=1).sum()

                top_1_robust_count, top_1_robustness = top_1_accuracy(
                    inf_count, golden_labels, faulty_out_scores
                )
                top_5_robust_count, top_5_robustness = top_5_accuracy(
                    inf_count, golden_labels, faulty_out_scores
                )

                non_critical_count = top_1_robust_count - masked_count

                critical_count = inf_count - top_1_robust_count

                report_data = [
                    inj_id,
                    target_layer_name,
                    weight_coords,
                    bit_pos,
                    inf_count,
                    top_1_count,
                    top_5_count,
                    top_1_robust_count,
                    top_5_robust_count,
                    masked_count,
                    non_critical_count,
                    critical_count,
                ]
                report_writer.writerow(report_data)
                if inj_num % 10 == 0:
                    f.flush()
                inf_manager.reset_faulty_run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform fault injections to weigths of a Tensorflow model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Dataset to use",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=512, help="Test set batch size"
    )
    parser.add_argument(
        "--network-name",
        "-n",
        type=str,
        required=True,
        help="Target network",
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--fault-list",
        "-f",
        type=str,
        required=False,
        help="Path to Fault list",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        required=False,
        help="Path to the generated output",
    )
    parser.add_argument(
        "--resume-from",
        "-r",
        type=int,
        required=False,
        default=0,
        help="Resume experiment from a certain injection id",
    )
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == "__main__":
    main(parse_args())
    # load_fault_list('test.csv')

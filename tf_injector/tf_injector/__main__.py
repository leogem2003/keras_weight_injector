import argparse

import tensorflow as tf  # type:ignore

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tf_injector.utils import SUPPORTED_MODELS, SUPPORTED_DATASETS
from tf_injector.loader import load_network
from tf_injector.injector import Injector
from tf_injector.metrics import gold_row_std_metric, make_faulty_row_std_metric


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
    parser.add_argument(
        "--save-scores",
        "-s",
        action="store_true",
        help="Save Injection Data",
    )
    parser.add_argument(
        "--sort-tf-layers",
        action="store_true",
        help="(Nat)Sort TF layers by name to make them match with their layer",
    )

    parsed_args = parser.parse_args()

    return parsed_args


def _pr(*args, end=""):
    print(*args, end=end, flush=True)


def main(args):
    network, dataset = load_network(args.network_name, args.dataset)
    injector = Injector(
        network,
        dataset,
        sort_layers=args.sort_tf_layers,
    )
    injector.load_fault_list(args.fault_list, resume_from=args.resume_from)

    injector.run_campaign(
        batch=args.batch_size,
        gold_row_metric=gold_row_std_metric,
        faulty_row_metric_maker=make_faulty_row_std_metric,
    )


if __name__ == "__main__":
    main(parse_args())

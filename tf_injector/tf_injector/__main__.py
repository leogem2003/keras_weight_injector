import argparse

import tensorflow as tf  # type:ignore

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tf_injector.utils import SUPPORTED_MODELS, SUPPORTED_DATASETS, DEFAULT_REPORT_DIR
from tf_injector.loader import load_network
from tf_injector.injector import Injector
from tf_injector.metrics import gold_row_std_metric, make_faulty_row_std_metric
from tf_injector.writer import CampaignWriter


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
        default=DEFAULT_REPORT_DIR,
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
        default=False,
        help="Save Injection Data",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="random seed for determinism"
    )

    parser.add_argument("--verbose", "-v", action="store_true")
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    if args.seed:
        tf.config.experimental.enable_op_determinism()
        tf.keras.utils.set_random_seed(args.seed)
        tf.keras.backend.manual_variable_initialization(True)

    network, dataset = load_network(args.network_name, args.dataset)
    injector = Injector(network, dataset)

    if args.fault_list is None:
        output, labels = injector.run_inference(args.batch_size)
        top_1, top_5 = gold_row_std_metric(output, labels)
        print(f"{top_1/len(dataset)*100:2.2f}")

        if args.save_scores:
            cw = CampaignWriter(args.dataset, args.network_name, args.output_path)
            cw.save_scores(output)
    else:
        injector.load_fault_list(args.fault_list, resume_from=args.resume_from)
        with CampaignWriter(args.dataset, args.network_name, args.output_path, timestamp=False) as cw:
            injector.run_campaign(
                batch=args.batch_size,
                save_scores=args.save_scores,
                gold_row_metric=gold_row_std_metric,
                faulty_row_metric_maker=make_faulty_row_std_metric,
                outputter=cw,
            )


if __name__ == "__main__":
    main(parse_args())

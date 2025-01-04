import argparse

import tensorflow as tf  # type:ignore

from tf_injector.inspector import Inspector, diff_composite
from tf_injector.utils import SUPPORTED_MODELS, SUPPORTED_DATASETS, DEFAULT_REPORT_DIR
from tf_injector.loader import load_network, dt_to_np
from tf_injector.injector import Injector
from tf_injector.metrics import gold_row_std_metric, make_faulty_row_std_metric
from tf_injector.writer import CampaignWriter

import sys

sys.setrecursionlimit(10**5)


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

    parser.add_argument("--inspect", action="store_true")

    parsed_args = parser.parse_args()

    return parsed_args

def inspect_layers(inspector, network, dt):
    d1, d2 = dt
    np1 = dt_to_np(d1)[:512]
    np2 = dt_to_np(d2)[:512]
    res1 = network(np1).numpy()
    res2 = network(np2).numpy()
    eq_idxs = (res1 == res2).any(axis=1)
    print("found %d different results" % len(eq_idxs))
    c1 = np1[eq_idxs][:1]
    c2 = np2[eq_idxs][:1]
    analysis = inspector.compare(c1, c2, diff_composite)
    for r in analysis:
        print(r)

import numpy as np

def mapping(layers, output):
    return list(zip(layers, output))

def test_noise(inspector, dt):
    d1, d2 = dt
    np1, gt1 = dt_to_np(d1)
    np2, _ = dt_to_np(d2)

    np1 = np1[:512]
    np2 = np2[:512]
    gt1 = gt1[:512]

    noise = np1-np2
    mask = (noise!=0).any(axis=1).any(axis=1).any(axis=1)
    img1 = np1[mask][1]
    noise = noise[mask][1]
    imgl1 = gt1[mask][1]
    print(img1.shape, imgl1.shape, noise.shape)

    noisy = [img1]
    for _ in range(10):
        noisy.append(img1+noise)
        noise*=2

    dt_noisy = np.array(noisy)
    labels = np.array([imgl1]*11)
    dt_noisy = tf.data.Dataset.from_tensor_slices((noisy, labels))
    dt_clean = tf.data.Dataset.from_tensor_slices((noisy[:1], labels[:1]))

    injector = Injector(inspector._model, dt_noisy)
    result_noisy = injector.run_inference(1)[0]

    injector = Injector(inspector._model, dt_clean)
    result_clean = injector.run_inference(1)[0]

    result_noisy = [
        mapping(inspector.outputs, r) for r in result_noisy
    ]

    result_clean = mapping(inspector.outputs, result_clean[0])
    stats = []
    for r in result_noisy:
        stats.append([(layer, diff_composite(v1, v2))
                      for (layer, v1), (_, v2) in zip(r, result_clean)])

    for j in range(len(stats[0])):
        print("\n", stats[0][j][0]) #layer
        for i in range(len(stats)):
            for num in stats[i][j][1][0]:
                if not isinstance(num, float):
                    num = float(num)
                str_num = f"{num:1.3}"
                print(f"{str_num:12}", end='')
            print()


def main(args):
    if args.seed:
        tf.config.experimental.enable_op_determinism()
        tf.keras.utils.set_random_seed(args.seed)
        tf.keras.backend.manual_variable_initialization(True)

    network, dataset = load_network(
        args.network_name, args.dataset, use_np=args.inspect
    )
    if args.inspect:
        inspector = Inspector(network)
        # inspect_layers(inspector, network, dataset)
        test_noise(inspector, dataset)
        return

    injector = Injector(network, dataset)
    if args.fault_list is None:
        output, labels = injector.run_inference(args.batch_size)
        top_1, top_5 = gold_row_std_metric(output, labels)
        print(
            f"GOLD stats:\nimages: {len(dataset)}\ntop 1 accuracy: {top_1}\ntop 5 accuracy: {top_5}"
        )
        if args.save_scores:
            cw = CampaignWriter(args.dataset, args.network_name, args.output_path)
            cw.save_scores(output)
    else:
        injector.load_fault_list(args.fault_list, resume_from=args.resume_from)
        with CampaignWriter(args.dataset, args.network_name, args.output_path) as cw:
            injector.run_campaign(
                batch=args.batch_size,
                save_scores=args.save_scores,
                gold_row_metric=gold_row_std_metric,
                faulty_row_metric_maker=make_faulty_row_std_metric,
                outputter=cw,
            )


if __name__ == "__main__":
    main(parse_args())

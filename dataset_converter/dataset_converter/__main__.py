import argparse
from dataset_converter.utils import get_loader, SUPPORTED_DATASETS
from dataset_converter.converter import save_tf_dataset


def main(args):
    _, loader = get_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        permute_tf=True,
    )

    next(iter(loader))
    save_tf_dataset(loader, args.dataset, args.output_path)


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
        "--output-path",
        "-o",
        type=str,
        required=False,
        help="Path to the generated output",
    )
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == "__main__":
    main(parse_args())

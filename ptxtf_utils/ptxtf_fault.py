import argparse
import csv
from typing import Callable, TypeVar


def create_matcher(file, rtl=True) -> dict[str, str]:
    matcher = {}
    for line in file.readlines():
        pt, tf = line.split(",")
        tf = tf.strip()
        if rtl:  # tf->pt
            pt, tf = tf, pt
        matcher[pt] = tf
    return matcher


def translate_fault(
    reader, writer, layer_matcher: dict[str, str], bit_permutation: Callable
):
    writer.writerow(next(iter(reader)))  # header
    for row in reader:
        id, layer, str_coords, bit = row
        layer = layer_matcher[layer]
        coords = tuple((int(coord) for coord in str_coords[1:-1].split(",")))
        coords = bit_permutation(coords)
        writer.writerow((id, layer, coords, bit))


CoordT = TypeVar("CoordT", tuple[int, int], tuple[int, int, int, int])


def permuter(coords: CoordT) -> CoordT:
    # permute(2, 1, 0)
    if len(coords) == 2:
        permuted = (coords[1], coords[0])
    elif len(coords) == 4:
        permuted = (coords[3], coords[2], coords[0], coords[1])
    else:
        raise ValueError(
            f"unsupported coordinate format: expected 2D or 4D, got {len(coords)}D"
        )
    return permuted


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="path to a csv fault list")
    parser.add_argument(
        "--pt",
        "-p",
        action="store_true",
        help="if set, translates a tf faultlist to a pt one",
    )

    parser.add_argument(
        "--layers", "-l", help="file containing pt and tf layers matching"
    )
    parser.add_argument("--output", "-o", help="output file path", default="./out.csv")
    return parser.parse_args()


def main(args):
    with open(args.layers, "r") as f:
        matcher = create_matcher(f, args.pt)

    with open(args.file_path, "r") as fr:
        with open(args.output, "w") as fw:
            translate_fault(csv.reader(fr), csv.writer(fw), matcher, permuter)


if __name__ == "__main__":
    main(parse_args())

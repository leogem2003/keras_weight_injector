import glob
import numpy as np
from itertools import chain
import os
import sys


def get_dirs(dir):
    return (d.__fspath__() for d in os.scandir(dir) if d.is_dir())


def test_all(base_dir):
    datasets = get_dirs(base_dir)
    models = {dataset: get_dirs(dataset) for dataset in datasets}
    for dt in models:
        for model in models[dt]:
            test(model)


def test(model):
    output_files = chain.from_iterable(
        (glob.glob(dir + "/*.npy") for dir in get_dirs(model))
    )
    clusters = {}
    for file in output_files:
        basename = os.path.basename(file)
        if not basename in clusters:
            clusters[basename] = []
        clusters[basename].append(file)

    for _, files in sorted(clusters.items()):
        if len(files) > 1:
            ref = np.load(files[0])
            for other in files[1:]:
                comp = np.load(other)
                try:
                    assert (ref == comp).all()
                except AssertionError:
                    distance = np.sqrt(((ref - comp) ** 2).sum())
                    print(
                        f"AssertionError: {files[0]} vs {other} differ with distance {distance:.6f}"
                    )


if __name__ == "__main__":
    test(sys.argv[1])

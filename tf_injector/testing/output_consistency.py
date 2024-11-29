import glob
import numpy as np
from itertools import chain
import os
import sys


def get_dirs(dir):
    yield from (d.__fspath__() for d in os.scandir(dir) if d.is_dir())


def create_batches(base_dir):
    datasets = get_dirs(base_dir)
    models = {dataset: get_dirs(dataset) for dataset in datasets}
    for dt in models:
        print("validating dataset", dt)
        for model in models[dt]:
            print("\tvalidating model", model)
            output_files = chain.from_iterable(
                (glob.glob(dir + "/*.npy") for dir in get_dirs(model))
            )
            clusters = {}
            for file in output_files:
                basename = os.path.basename(file)
                if not basename in clusters:
                    clusters[basename] = []
                clusters[basename].append(file)

            for cl, files in sorted(clusters.items()):
                print(f"\t\tvalidating {cl} using reference {files[0]}")
                if len(files) > 1:
                    ref = np.load(files[0])
                    for other in files[1:]:
                        comp = np.load(other)
                        try:
                            assert (ref == comp).all()
                            print(f"\t\t\t{other:>32} OK")
                        except AssertionError:
                            print(f">>>>\t\t{other:>32} AssertionError!")


if __name__ == "__main__":
    create_batches(sys.argv[1])

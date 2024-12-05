import glob
import os
import sys
import csv


def get_dirs(dir):
    yield from (d.__fspath__() for d in os.scandir(dir) if d.is_dir())


def get_length(file):
    with open(file, "r") as f:
        return len(list(csv.reader(f)))


def validate_reports(base_dir):
    datasets = get_dirs(base_dir)
    models = {dataset: get_dirs(dataset) for dataset in datasets}
    for dt in models:
        for model in models[dt]:
            validate_model(model)


def validate_model(model):
    csvs = sorted(glob.glob(model + "/*.csv"))
    injs = []
    all_data = {}
    for file in csvs:
        with open(file, "r") as f:
            reader = csv.reader(f)
            all_data[file] = []
            for row in reader:
                if row[:4] not in injs:
                    injs.append(row[:4])
                all_data[file].append(row)
    for inj in injs:
        selected = []
        for file, finjs in all_data.items():
            for fi in finjs:
                if inj == fi[:4]:
                    selected.append((file, fi))
                    break
        first = selected[0][1]
        for file, i in selected:
            try:
                assert i == first
            except AssertionError:
                print(
                    f"{model}: Assertion Error {selected[0][0]} vs {file}:\
{first[4:]} vs {i[4:]}"
                )


if __name__ == "__main__":
    validate_model(sys.argv[1])

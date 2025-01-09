import os
import glob
import csv
import numpy as np

def get_dirs(dir):
    return (d.__fspath__() for d in os.scandir(dir) if d.is_dir())


def get_networks(dir):
    networks = []
    for dataset in sorted(get_dirs(dir)):
        networks.append((os.path.basename(dataset), [
            os.path.basename(file).split(".")[0]
            for file in sorted(glob.glob(dataset + "/*.keras"))
        ]))
    return networks


def cmd(*args, catch=False):
    result = os.system(" ".join(args))
    if result != 0 and catch:
        raise RuntimeError(f"{cmd}: system exited with status {result}")


def run(models):
    cmd("echo 'dataset, model, top1, top5, top1_robust, top5_robust, masked, non_critical, critical'",
        ">> faulty_report.txt")
    for dt, networks in models:
        for net in networks:
            print("running", dt, net)
            cmd("cd ../tf_injector &&",
                f"python -m tf_injector -d {dt} -n {net}",
                f"-f fault_lists/{dt}/{net}_TF_FL.csv -o ../test/reports/")

            with open(f"reports/{dt}/{net}/{dt}_{net}.csv", "r") as f, open("faulty_report.txt", "a") as frep:
                reader = csv.reader(f)
                fields = next(iter(reader))[5:]
                next(iter(reader)) #gold row
                sums = [0.0]*len(fields)
                n_rows = 0
                for row in reader:
                    n_rows += 1
                    row = row[4:]
                    for idx, field in enumerate(row[1:]):
                        sums[idx] += int(field)/int(row[0])

                sums = ",".join((f"{sum/n_rows*100:3.2f}" for sum in sums))
                frep.write(f"{dt},{net},{sums}\n")



def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf logs
    models = get_networks("../tf_injector/models/")
    models = [models[0]]
    run(models)


if __name__ == "__main__":
    main()

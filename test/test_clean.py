import glob
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf logs

def get_dirs(dir):
    return (d.__fspath__() for d in os.scandir(dir) if d.is_dir())


def get_networks(dir):
    datasets = get_dirs(dir)
    networks = {}
    for dataset in datasets:
        networks[os.path.basename(dataset)] = [
            os.path.basename(file).split(".")[0]
            for file in sorted(glob.glob(dataset + "/*.keras"))
        ]
    return networks

def cmd(*args, catch=False):
    command = " ".join(args)
    status = os.system(command)
    if catch and status != 0:
        raise RuntimeError(f"{command}: Bad exit status {status}")

def run(models):
    for dt in models:
        for model in models[dt]:
            cmd(f"echo -n '{dt:8} {model:12}' >> clean_report.txt")
            print("testing", dt, model)
            cmd("cd ../tf_injector &&",
                f"python -m tf_injector -d {dt} -n {model} -b 128 >> ../test/clean_report.txt", catch=True)

if __name__ == "__main__":
    dir = "../tf_injector/models/"
    models = get_networks(dir)
    run(models)

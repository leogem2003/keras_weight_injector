import os
import glob
from pathlib import Path
from itertools import chain
import sys
import importlib.util


def get_dirs(dir):
    return (d.__fspath__() for d in os.scandir(dir) if d.is_dir())


def import_func(path, *functions):
    spec = importlib.util.spec_from_file_location("pt_network", path)
    if spec is None:
        raise FileNotFoundError(f"Cannot find file {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["pt_network"] = module

    if spec.loader is None:
        raise ValueError("Cannot load module")

    spec.loader.exec_module(module)
    funcs = []
    for funcname in functions:
        funcs.append(getattr(module, funcname))

    return funcs


def logger(msg):
    def decorator(f):
        def wrapper(*args, **kwargs):
            print(f"{msg}...", end=" ", flush=True)
            res = f(*args, **kwargs)
            print("done", flush=True)
            return res

        return wrapper

    return decorator


@logger("collecting networks")
def get_networks(dir):
    datasets = get_dirs(dir)
    networks = {}
    for dataset in datasets:
        networks[dataset] = glob.glob(dataset + "/*.keras")

    return networks


imported_modules = {}


def py_call(path, cmd):
    if not path in imported_modules:
        main, parser = import_func(path, "main", "parse_args")

        def call(args):
            return main(parser(args))

        imported_modules[path] = call

    imported_modules[path](cmd.split(" "))


@logger("generating fault lists")
def gen_faultlists(networks):
    pt_dt = list(get_dirs("../legacy_benchmark_models/benchmark_models/models/"))
    pt_files = list(chain.from_iterable((glob.glob(path + "/*.py") for path in pt_dt)))

    for dt, nets in networks.items():
        dt_name = os.path.basename(dt)
        os.makedirs("./faultlists/" + dt_name, exist_ok=True)
        print("processing", dt)
        for net in nets:
            netname = Path(net).stem
            prefix = netname.lower()[:3]
            pt_file = None
            for pt_py in pt_files:
                dir, filename = os.path.split(pt_py)
                if dir.endswith(dt_name) and prefix in filename:
                    pt_file = pt_py
                    break

            if not pt_file:
                raise FileNotFoundError(f"cannot find PT .py for {net}")

            print("network:", netname)
            prefix = f"faultlists/{dt_name}/{netname}"
            ### LAYER MATCHING ###
            print(netname, "layer matching")
            py_call(
                "../ptxtf_utils/ptxtf_net.py",
                f"{pt_file} {netname} {net} -o {prefix}_match",
            )

            ### FAULTLIST ###
            print(netname, "tf fault list")
            py_call(
                "../ptxtf_utils/fault_writer.py",
                f"-n 4 -s 0 -o {prefix}_faultTF.csv tf {net}",
            )

            print(netname, "pt fault list")
            py_call(
                "../ptxtf_utils/ptxtf_fault.py",
                f"{prefix}_faultTF.csv --pt -l {prefix}_match -o {prefix}_faultPT.csv",
            )


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable cuda
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf logs
    networks = get_networks("../tf_injector/models/")
    gen_faultlists(networks)


if __name__ == "__main__":
    main()

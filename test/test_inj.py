import os
import glob


def get_dirs(dir):
    return (d.__fspath__() for d in os.scandir(dir) if d.is_dir())


def shell(*cmd, catch_error=False):
    result = os.system(" ".join(cmd))
    if result != 0 and catch_error:
        raise RuntimeError(f"system exited with status {result}")


def get_models(path, target_models=None):
    datasets = get_dirs(path)
    models = {}
    # models structure: dataset:[(model, tf_fault, pt_fault),...]
    for dt in datasets:
        dt_name = os.path.basename(dt)
        models[dt_name] = []
        for match, tf_fault, pt_fault in zip(
            sorted(glob.glob(dt + "/*_match")),
            sorted(glob.glob(dt + "/*_faultTF.csv")),
            sorted(glob.glob(dt + "/*_faultPT.csv")),
        ):
            models[dt_name].append(
                (
                    os.path.basename(match).replace("_match", ""),
                    os.path.join("../test/", tf_fault),
                    pt_fault,
                )
            )
    if target_models:
        for dt, dt_models in models.items():
            for model in dt_models:
                if model[0] == target_models[dt]:
                    models[dt] = [model]
    return models


def run(models, dt=None, clean=False):
    if dt:
        tasks = [(dt, models[dt])]
    else:
        tasks = models.items()

    legacy_injector = (
        "../legacy_benchmark_models/benchmark_models/injector/tf_weight_injection.py"
    )

    tf_injector = "-m tf_injector"
    for dt, models in tasks:
        for model, tf_fault, pt_fault in models:
            print(f"testing {model}_{dt}")
            print("Running legacy version")
            shell(
                "python",
                legacy_injector,
                f"-d {dt} -n {model} -f {pt_fault} -s -o ./reports/{dt}/{model}/{dt}_{model}_legacy.csv",
            )

            print("Running new version")
            if clean:
                shell(
                    "cd ../tf_injector &&",
                    "python",
                    tf_injector,
                    f"-d {dt} -n {model} -s -o ../test/reports",
                    "&& cd ../test",
                )
            else:
                shell(
                    "cd ../tf_injector &&",
                    "python",
                    tf_injector,
                    f"-d {dt} -n {model} -f {tf_fault} -s -o ../test/reports",
                    "&& cd ../test",
                )

            print("Comparing")
            shell(
                "python",
                "../tf_injector/testing/output_consistency.py",
                f"./reports/{dt}/{model}/",
            )
            shell(
                "python",
                "../ptxtf_utils/ptxtf_fault.py",
                f"./reports/{dt}/{model}/{dt}_{model}_legacy.csv",
                f"-l ./faultlists/{dt}/{model}_match",
                f"-o ./reports/{dt}/{model}/{dt}_{model}_legacyTF.csv",
                "--from-report",
                "--no-permute",  # report has PT layer but TF coords!
            )
            shell(
                "python",
                "../tf_injector/testing/reports_consistency.py",
                f"./reports/{dt}/{model}/",
            )
            print("Done\n")


def main():
    models = {
        "CIFAR10": "MobileNetV2",
        "CIFAR100": "ResNet18",
        "GTSRB": "ResNet20",
    }
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf logs
    models = get_models("./faultlists/", models)

    run(models, clean=True)


if __name__ == "__main__":
    main()

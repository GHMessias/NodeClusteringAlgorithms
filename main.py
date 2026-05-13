import argparse
import glob
import os
import subprocess
import sys
import time


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

MODELS = {
    "coane": {
        "script": os.path.join(
            MODELS_DIR,
            "CoANE",
            "CoANE with attributes preservation",
            "main_CoANE.py",
        ),
    },
    "coane_topk": {
        "script": os.path.join(
            MODELS_DIR,
            "CoANE",
            "CoANE topk with attributes preservation",
            "main_CoANE_topk.py",
        ),
    },
    "dgc_efr": {
        "root": os.path.join(MODELS_DIR, "DGC-EFR"),
        "preae": os.path.join(MODELS_DIR, "DGC-EFR", "preae.py"),
        "pregae": os.path.join(MODELS_DIR, "DGC-EFR", "pregae.py"),
        "train": os.path.join(MODELS_DIR, "DGC-EFR", "dgc_efr.py"),
    },
    "soli": {
        "script": os.path.join(MODELS_DIR, "SOLI", "SOLI_GAE", "main.py"),
    },
    "sanece": {
        "script": os.path.join(MODELS_DIR, "SANECE", "main.py"),
    },
    "egc_vgae": {
        "script": os.path.join(MODELS_DIR, "EG-VGAE", "EGC_VGAE.py"),
    },
    "ucode": {
        "script": os.path.join(MODELS_DIR, "UCODE", "main.py"),
    },
    "ns4gc": {
        "script": os.path.join(MODELS_DIR, "NS4GC", "main.py"),
    },
    "dgcluster": {
        "script": os.path.join(MODELS_DIR, "DGCluster", "main.py"),
    },
    "dsagc": {
        "script": os.path.join(MODELS_DIR, "DSAGC", "main.py"),
    },
    "scgc": {
        "script": os.path.join(MODELS_DIR, "SCGC", "train.py"),
    },
}

ALL_MODELS = [
    "coane",
    "coane_topk",
    "dgc_efr",
    "soli",
    "ucode",
    "ns4gc",
    "dgcluster",
    "sanece",
    "egc_vgae",
    "dsagc",
    "scgc",
]


def abs_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.abspath(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Algorithms clustering benchmark runner")
    parser.add_argument("--model", required=True, choices=sorted([*MODELS, "all"]), help="Algorithm to run")
    parser.add_argument("--data_pt", type=str, default=None, help="Optional PyG data.pt path")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional dataset directory")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for labels output")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--device", type=str, default="cpu", help='Training device, for example "cpu" or "cuda"')
    parser.add_argument("--kmeans_clusters", type=int, default=None, help="Optional KMeans cluster count")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Keep running remaining models when one command fails",
    )
    return parser.parse_known_args()


def build_coane_command(args, extra_args):
    script = MODELS[args.model]["script"]
    output_dir = abs_path(args.output_dir)

    command = [
        sys.executable,
        script,
        "--dataset",
        args.dataset,
        "--device",
        args.device,
        "--output_dir",
        output_dir,
        "--kmeans",
    ]

    if args.data_pt:
        command.extend(["--data_pt", abs_path(args.data_pt)])
    if args.data_dir:
        command.extend(["--data_dir", abs_path(args.data_dir)])
    if args.epochs is not None:
        command.extend(["--num_epochs", str(args.epochs)])
    if args.kmeans_clusters is not None:
        command.extend(["--kmeans_clusters", str(args.kmeans_clusters)])

    command.extend(extra_args)
    return command, os.path.dirname(script)


def dgc_common_args(args):
    common = ["--name", args.dataset]
    if args.data_pt:
        common.extend(["--data_pt", abs_path(args.data_pt)])
    if args.epochs is not None:
        common.extend(["--epochs", str(args.epochs)])
    if args.kmeans_clusters is not None:
        common.extend(["--n_clusters", str(args.kmeans_clusters)])
    return common


def build_dgc_efr_commands(args, extra_args):
    model = MODELS["dgc_efr"]
    output_dir = abs_path(args.output_dir)
    pretrain_dir = os.path.join(output_dir, "dgc_efr_pretrain")
    os.makedirs(pretrain_dir, exist_ok=True)

    preae_path = os.path.join(pretrain_dir, "preae_{}.pkl".format(args.dataset))
    pregae_path = os.path.join(pretrain_dir, "pregae_{}.pkl".format(args.dataset))
    common = dgc_common_args(args)

    commands = [
        [
            sys.executable,
            model["preae"],
            *common,
            "--preae_path",
            preae_path,
        ],
        [
            sys.executable,
            model["pregae"],
            *common,
            "--pregae_path",
            pregae_path,
        ],
        [
            sys.executable,
            model["train"],
            *common,
            "--output_dir",
            output_dir,
            "--preae_path",
            preae_path,
            "--pregae_path",
            pregae_path,
            *extra_args,
        ],
    ]
    return commands, model["root"]


def build_soli_command(args, extra_args):
    script = MODELS["soli"]["script"]
    command = [
        sys.executable,
        script,
        "--dataset",
        args.dataset,
        "--device",
        args.device,
        "--output_dir",
        abs_path(args.output_dir),
    ]

    if args.data_pt:
        command.extend(["--data_pt", abs_path(args.data_pt)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.kmeans_clusters is not None:
        command.extend(["--kmeans_clusters", str(args.kmeans_clusters)])

    command.extend(extra_args)
    return command, os.path.dirname(script)


def build_sanece_command(args, extra_args):
    script = MODELS["sanece"]["script"]
    command = [
        sys.executable,
        script,
        "--dataset",
        args.dataset,
        "--output_dir",
        abs_path(args.output_dir),
    ]

    if args.data_pt:
        command.extend(["--data_pt", abs_path(args.data_pt)])
    if args.epochs is not None:
        command.extend(["--epoch", str(args.epochs)])
    if args.kmeans_clusters is not None:
        command.extend(["--kmeans_clusters", str(args.kmeans_clusters)])

    command.extend(extra_args)
    return command, os.path.dirname(script)


def build_egc_vgae_command(args, extra_args):
    script = MODELS["egc_vgae"]["script"]
    command = [
        sys.executable,
        script,
        "--dataset",
        args.dataset,
        "--output_dir",
        abs_path(args.output_dir),
    ]

    if args.data_pt:
        command.extend(["--data_pt", abs_path(args.data_pt)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])

    command.extend(extra_args)
    return command, os.path.dirname(script)


def build_standard_command(args, extra_args):
    script = MODELS[args.model]["script"]
    command = [
        sys.executable,
        script,
        "--output_dir",
        abs_path(args.output_dir),
    ]

    if args.data_pt:
        command.extend(["--data_pt", abs_path(args.data_pt)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.device:
        command.extend(["--device", args.device])
    if args.kmeans_clusters is not None and args.model in {"ucode", "ns4gc"}:
        command.extend(["--kmeans_clusters", str(args.kmeans_clusters)])
    if args.kmeans_clusters is not None and args.model == "dsagc":
        command.extend(["--n_clusters", str(args.kmeans_clusters)])

    command.extend(extra_args)
    return command, os.path.dirname(script)


def build_scgc_command(args, extra_args):
    script = MODELS["scgc"]["script"]
    command = [
        sys.executable,
        script,
        "--output_dir",
        abs_path(args.output_dir),
        "--device",
        args.device,
        "--model",
        "SCGC",
        "--iterations",
        "1",
    ]

    if args.data_pt:
        command.extend(["--data_pt", abs_path(args.data_pt)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.kmeans_clusters is not None:
        command.extend(["--n_clusters", str(args.kmeans_clusters)])

    command.extend(extra_args)
    return command, os.path.dirname(script)


def subprocess_env(args):
    env = os.environ.copy()
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def recent_labels(output_dir, started_at):
    paths = glob.glob(os.path.join(abs_path(output_dir), "*_labels.npy"))
    return sorted(path for path in paths if os.path.getmtime(path) >= started_at)


def build_model_commands(args, extra_args):
    if args.model == "dgc_efr":
        return build_dgc_efr_commands(args, extra_args)
    if args.model == "soli":
        command, cwd = build_soli_command(args, extra_args)
        return [command], cwd
    if args.model == "sanece":
        command, cwd = build_sanece_command(args, extra_args)
        return [command], cwd
    if args.model == "egc_vgae":
        command, cwd = build_egc_vgae_command(args, extra_args)
        return [command], cwd
    if args.model == "scgc":
        command, cwd = build_scgc_command(args, extra_args)
        return [command], cwd
    if args.model in {"ucode", "ns4gc", "dgcluster", "dsagc"}:
        command, cwd = build_standard_command(args, extra_args)
        return [command], cwd

    command, cwd = build_coane_command(args, extra_args)
    return [command], cwd


def run_one_model(model_name, args, extra_args, env):
    model_args = argparse.Namespace(**vars(args))
    model_args.model = model_name
    started_at = time.time()
    commands, cwd = build_model_commands(model_args, extra_args)

    for command in commands:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=cwd, check=True, env=env)

    outputs = recent_labels(model_args.output_dir, started_at)
    return outputs


def print_summary(results):
    print("\nBenchmark summary:")
    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        print(f"{status:4} {result['model']}")
        if result["outputs"]:
            for output in result["outputs"]:
                print(f"     {output}")
        if result["error"]:
            print(f"     {result['error']}")


def main():
    args, extra_args = parse_args()
    env = subprocess_env(args)
    models_to_run = ALL_MODELS if args.model == "all" else [args.model]
    results = []

    for model_name in models_to_run:
        try:
            outputs = run_one_model(model_name, args, extra_args, env)
            results.append({"model": model_name, "ok": True, "outputs": outputs, "error": None})
        except subprocess.CalledProcessError as exc:
            error = f"exit status {exc.returncode}: {' '.join(exc.cmd)}"
            results.append({"model": model_name, "ok": False, "outputs": [], "error": error})
            if not args.continue_on_error:
                print_summary(results)
                raise

    print_summary(results)


if __name__ == "__main__":
    main()

import argparse
import glob
import json
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

MODEL_OUTPUT_NAMES = {
    "coane": "CoANE",
    "coane_topk": "CoANE_topk",
    "dgc_efr": "DGC",
    "soli": "SOLI",
    "sanece": "SANECE",
    "egc_vgae": "EGC_VGAE",
    "ucode": "UCODE",
    "ns4gc": "NS4GC",
    "dgcluster": "DGCluster",
    "dsagc": "DSAGC",
    "scgc": "SCGC",
}

RUN_STATUS_FILENAME = "run_status.json"
RUNNER_LOGS_DIRNAME = "runner_logs"
ERROR_TAIL_LINES = 20


def abs_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.abspath(path)


def model_device_arg(model_name, device):
    if model_name == "dgcluster" and device == "cuda":
        return "cuda:0"
    return device


def model_output_name(model_name):
    return MODEL_OUTPUT_NAMES.get(model_name, model_name)


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
    if args.device:
        command.extend(["--device", args.device])
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
    device = model_device_arg(args.model, args.device)
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
    if device:
        command.extend(["--device", device])
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


def runner_logs_dir(output_dir):
    return os.path.join(abs_path(output_dir), RUNNER_LOGS_DIRNAME)


def model_log_path(output_dir, model_name):
    return os.path.join(runner_logs_dir(output_dir), "{}.log".format(model_output_name(model_name)))


def run_status_path(output_dir):
    return os.path.join(abs_path(output_dir), RUN_STATUS_FILENAME)


def ensure_output_layout(output_dir):
    os.makedirs(abs_path(output_dir), exist_ok=True)
    os.makedirs(runner_logs_dir(output_dir), exist_ok=True)


def tail_nonempty_lines(text, limit=ERROR_TAIL_LINES):
    if not text:
        return ""

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-limit:])


def run_command_logged(command, cwd, env, log_file, append=False):
    mode = "a" if append else "w"
    command_text = " ".join(command)
    output_lines = []

    with open(log_file, mode, encoding="utf-8") as handle:
        handle.write("$ {}\n".format(command_text))
        handle.flush()

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            output_lines.append(line)

        returncode = process.wait()
        handle.write("\n[exit_status={}]\n\n".format(returncode))

    output = "".join(output_lines)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, output=output)

    return output


def save_run_status(output_dir, results):
    ensure_output_layout(output_dir)
    payload = {"results": results}
    with open(run_status_path(output_dir), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_true_labels_for_metrics(data_pt):
    from evaluate_outputs import load_true_labels

    return load_true_labels(abs_path(data_pt))


def evaluate_output_metrics(outputs, y_true):
    from evaluate_outputs import evaluate_file

    metrics = []
    errors = []

    for output in outputs:
        try:
            row = evaluate_file(output, y_true)
            metrics.append(
                {
                    "file": output,
                    "acc": row["acc"],
                    "f1": row["f1_macro"],
                    "nmi": row["nmi"],
                }
            )
        except Exception as exc:
            errors.append("{}: {}".format(output, exc))

    return metrics, errors


def format_metric_value(value):
    return "{:.4f}".format(value)


def summarize_process_error(exc):
    command_text = " ".join(exc.cmd)
    message = "exit status {}: {}".format(exc.returncode, command_text)
    output_tail = tail_nonempty_lines(exc.output)
    if output_tail:
        return "{}\n{}".format(message, output_tail)
    return message


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
    ensure_output_layout(model_args.output_dir)
    started_at = time.time()
    commands, cwd = build_model_commands(model_args, extra_args)
    log_file = model_log_path(model_args.output_dir, model_name)

    for index, command in enumerate(commands):
        print("Running:", " ".join(command), flush=True)
        run_command_logged(command, cwd=cwd, env=env, log_file=log_file, append=index > 0)

    outputs = recent_labels(model_args.output_dir, started_at)
    return {"outputs": outputs, "log_file": log_file}


def print_summary(results):
    print("\nBenchmark summary:")
    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        print(f"{status:4} {result['model']}")
        if result["outputs"]:
            for output in result["outputs"]:
                print(f"     {output}")
        if result["metrics"]:
            for metric in result["metrics"]:
                metric_file = os.path.basename(metric["file"])
                print(
                    "     metrics [{}] acc={} f1={} nmi={}".format(
                        metric_file,
                        format_metric_value(metric["acc"]),
                        format_metric_value(metric["f1"]),
                        format_metric_value(metric["nmi"]),
                    )
                )
        if result["metrics_error"]:
            for error in result["metrics_error"]:
                print(f"     metrics error: {error}")
        if result["error"]:
            print(f"     {result['error']}")
        if result["log_file"]:
            print(f"     log: {result['log_file']}")


def main():
    args, extra_args = parse_args()
    env = subprocess_env(args)
    models_to_run = ALL_MODELS if args.model == "all" else [args.model]
    results = []
    ensure_output_layout(args.output_dir)
    y_true = None
    metrics_setup_error = ""

    if args.data_pt:
        try:
            y_true = load_true_labels_for_metrics(args.data_pt)
        except Exception as exc:
            metrics_setup_error = "unable to load true labels from {}: {}".format(abs_path(args.data_pt), exc)

    for model_name in models_to_run:
        display_name = model_output_name(model_name)
        log_file = model_log_path(args.output_dir, model_name)
        try:
            run_result = run_one_model(model_name, args, extra_args, env)
            outputs = run_result["outputs"]
            error = ""
            ok = True
            metrics = []
            metrics_error = []
            if not outputs:
                ok = False
                error = "command completed but no *_labels.npy files were created"
            elif not args.data_pt:
                metrics_error = ["metrics not computed because --data_pt was not provided"]
            elif metrics_setup_error:
                metrics_error = [metrics_setup_error]
            elif y_true is not None:
                metrics, metrics_error = evaluate_output_metrics(outputs, y_true)

            results.append(
                {
                    "runner_model": model_name,
                    "model": display_name,
                    "ok": ok,
                    "outputs": outputs,
                    "metrics": metrics,
                    "metrics_error": metrics_error,
                    "error": error,
                    "log_file": run_result["log_file"],
                }
            )
        except subprocess.CalledProcessError as exc:
            results.append(
                {
                    "runner_model": model_name,
                    "model": display_name,
                    "ok": False,
                    "outputs": [],
                    "metrics": [],
                    "metrics_error": [],
                    "error": summarize_process_error(exc),
                    "log_file": log_file,
                }
            )
            save_run_status(args.output_dir, results)
            if not args.continue_on_error:
                print_summary(results)
                raise

        save_run_status(args.output_dir, results)

    print_summary(results)


if __name__ == "__main__":
    main()

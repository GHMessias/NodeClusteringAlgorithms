from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


CLASS_TO_MODELS = OrderedDict(
    {
        "Reformulation Model": ["CoANE", "CoANE_topk", "DSAGC", "SANECE"],
        "KL-Div Model": ["DGC"],
        "Topological Model": ["SOLI", "NS4GC"],
        "Optimization Model": ["DGCluster", "UCODE", "EGC_VGAE"],
    }
)

DISPLAY_NAME = {
    "CoANE": "CoANE",
    "CoANE_topk": "CoANE top-k",
    "DGC": "DGC-EFR",
    "DSAGC": "DSAGC",
    "DGCluster": "DGCluster",
    "EGC_VGAE": "EG-VGAE",
    "NS4GC": "NS4GC",
    "SANECE": "SANECE",
    "SOLI": "SOLI",
    "UCODE": "UCoDe",
}

MODEL_TO_CLASS = {
    model: taxonomy_class
    for taxonomy_class, models in CLASS_TO_MODELS.items()
    for model in models
}

METRIC_LABEL = {
    "acc": "ACC",
    "nmi": "NMI",
    "ari": "ARI",
    "f1_macro": "F1 Macro",
    "f1_micro": "F1 Micro",
}

SCENARIO_LABEL = {
    "heterophily": "Heterophily",
    "edges": "Edges",
    "comm_collapse": "Collapse level",
}


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def choose_x_column(df: pd.DataFrame) -> str:
    if "collapse_level" in df.columns and df["collapse_level"].notna().any():
        return "collapse_level"
    if "scenario_value" in df.columns and df["scenario_value"].notna().any():
        return "scenario_value"
    if "n_samples" in df.columns and df["n_samples"].notna().any():
        return "n_samples"
    raise ValueError("Could not infer the x-axis column from the CSV.")


def build_taxonomy_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    df["taxonomy_class"] = df["model"].map(MODEL_TO_CLASS)
    df["display_model"] = df["model"].map(DISPLAY_NAME).fillna(df["model"])
    return df


def export_plot_points(
    plot_df: pd.DataFrame,
    csv_path: Path,
    metric: str,
    x_col: str,
    x_label: str,
    class_order: list[str],
    output_dir: Path,
) -> Path:
    rows: list[dict[str, object]] = []

    for panel_index, taxonomy_class in enumerate(class_order, start=1):
        class_df = plot_df[plot_df["taxonomy_class"] == taxonomy_class].copy()
        for model_name in class_df["model"].drop_duplicates():
            model_df = class_df[class_df["model"] == model_name].sort_values(x_col)
            for _, row in model_df.iterrows():
                rows.append(
                    {
                        "panel_index": panel_index,
                        "taxonomy_class": taxonomy_class,
                        "taxonomy_slug": slugify(taxonomy_class),
                        "model": model_name,
                        "display_model": row["display_model"],
                        "x": row[x_col],
                        "y": row[metric],
                    }
                )

    points_df = pd.DataFrame(rows)
    txt_path = output_dir / f"{csv_path.stem}_{metric}_taxonomy_points.txt"

    with txt_path.open("w", encoding="utf-8") as fp:
        fp.write(f"# source_csv: {csv_path}\n")
        fp.write(f"# metric: {metric}\n")
        fp.write(f"# x_column: {x_col}\n")
        fp.write(f"# x_label: {x_label}\n")
        fp.write(
            "# columns: panel_index taxonomy_class taxonomy_slug model display_model x y\n"
        )
        points_df.to_csv(fp, sep="\t", index=False)

    return txt_path


def plot_one_csv(
    csv_path: Path,
    output_dir: Path,
    metrics: list[str],
    include_unclassified: bool,
) -> None:
    df = build_taxonomy_frame(csv_path)
    class_to_models = OrderedDict(CLASS_TO_MODELS)

    unclassified = sorted(df.loc[df["taxonomy_class"].isna(), "model"].dropna().unique())
    if unclassified:
        print(
            f"[{csv_path.name}] models outside the LaTeX taxonomy: "
            + ", ".join(unclassified)
        )

    if not include_unclassified:
        df = df[df["taxonomy_class"].notna()].copy()
    else:
        df["taxonomy_class"] = df["taxonomy_class"].fillna("Unclassified")
        if unclassified:
            class_to_models["Unclassified"] = unclassified

    if df.empty:
        print(f"[{csv_path.name}] no rows left after taxonomy filtering.")
        return

    x_col = choose_x_column(df)
    scenario = df["scenario"].dropna().iloc[0] if "scenario" in df.columns else ""
    x_label = SCENARIO_LABEL.get(scenario, x_col.replace("_", " ").title())

    class_order = [
        taxonomy_class
        for taxonomy_class in class_to_models
        if (df["taxonomy_class"] == taxonomy_class).any()
    ]

    ncols = 2
    nrows = math.ceil(len(class_order) / ncols)

    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        if metric not in df.columns:
            print(f"[{csv_path.name}] metric '{metric}' not found, skipping.")
            continue

        plot_df = df[df[metric].notna()].copy()
        if plot_df.empty:
            print(f"[{csv_path.name}] metric '{metric}' has no valid rows, skipping.")
            continue

        txt_path = export_plot_points(
            plot_df=plot_df,
            csv_path=csv_path,
            metric=metric,
            x_col=x_col,
            x_label=x_label,
            class_order=class_order,
            output_dir=output_dir,
        )

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(7 * ncols, 4.8 * nrows),
            sharey=True,
        )
        axes = pd.Series(axes.ravel() if hasattr(axes, "ravel") else [axes])

        for ax, taxonomy_class in zip(axes, class_order):
            class_df = plot_df[plot_df["taxonomy_class"] == taxonomy_class].copy()
            for model_name in class_df["model"].drop_duplicates():
                model_df = class_df[class_df["model"] == model_name].sort_values(x_col)
                ax.plot(
                    model_df[x_col],
                    model_df[metric],
                    marker="o",
                    linewidth=2.2,
                    label=model_df["display_model"].iloc[0],
                )

            ax.set_title(taxonomy_class)
            ax.set_xlabel(x_label)
            ax.set_ylabel(METRIC_LABEL.get(metric, metric.upper()))
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best", fontsize=10, frameon=True)

        for ax in axes[len(class_order) :]:
            ax.axis("off")

        fig.suptitle(
            f"{csv_path.stem} | {METRIC_LABEL.get(metric, metric.upper())} by taxonomy",
            fontsize=18,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        out_path = output_dir / f"{csv_path.stem}_{metric}_taxonomy.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out_path}")
        print(f"saved: {txt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot benchmark CSVs grouped by taxonomy class."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("outputs/csvs"),
        help="Directory containing the benchmark CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/taxonomy_plots"),
        help="Directory where the plots will be saved.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["acc", "nmi", "ari", "f1_macro", "f1_micro"],
        help="Metrics to plot.",
    )
    parser.add_argument(
        "--include-unclassified",
        action="store_true",
        help="Also create an 'Unclassified' subplot for models missing from the LaTeX taxonomy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = sorted(args.csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args.csv_dir}")

    for csv_path in csv_paths:
        plot_one_csv(
            csv_path=csv_path,
            output_dir=args.output_dir,
            metrics=args.metrics,
            include_unclassified=args.include_unclassified,
        )


if __name__ == "__main__":
    main()

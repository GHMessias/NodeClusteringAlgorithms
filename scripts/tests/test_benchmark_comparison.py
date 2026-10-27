"""Comparison directions, incomplete experiments, retries, and colored exports."""

from pathlib import Path
from itertools import combinations
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_comparison as comparison


class BenchmarkComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_dir = self.root / "configs"
        for dataset in ("cora", "wiki"):
            for model in ("ucode", "ns4gc", "densgnn"):
                folder = self.config_dir / f"core_border_adaptive_{dataset}"
                folder.mkdir(parents=True, exist_ok=True)
                config = {
                    "experiment": {"dataset": f"{dataset}_benchmark_adaptive", "output_root": "results", "seed": 1, "runs": 2},
                    "output": {"csv_filename": "metrics.csv"},
                    "metrics": {"include": ["acc", "runtime_seconds", "n_clusters"]},
                    "models": {model: {"enabled": True, "params": {"point_selection": "adaptive"} if model == "densgnn" else {}}},
                }
                (folder / f"{model}.yaml").write_text(yaml.safe_dump(config))

    def save(self, dataset, model, seed, acc, run="01", status="success", error="", runtime=10,
             source_dataset=None, **extra_metrics):
        run_id = f"{run}_{model}_seed{seed}"
        source_dataset = source_dataset or f"{dataset}_benchmark_adaptive"
        path = self.root / "results" / source_dataset / "runs" / run_id / "metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "dataset": source_dataset, "model": model, "seed": seed,
            "run_id": run_id, "status": status, "error": error,
            "acc": acc, "runtime_seconds": runtime, "n_clusters": 2, **extra_metrics,
        }]).to_csv(path, index=False)
        return path

    def report(self, metrics=("acc", "runtime_seconds", "n_clusters")):
        return comparison.build_report(self.config_dir, self.root, metrics)

    def test_directions_ties_and_absent_reference(self):
        self.assertEqual(comparison.compare_mean(.8, .6, "acc"), "better")
        self.assertEqual(comparison.compare_mean(.4, .6, "acc"), "worse")
        self.assertEqual(comparison.compare_mean(5, 10, "runtime_seconds"), "better")
        self.assertEqual(comparison.compare_mean(2, 1, "davies_bouldin"), "worse")
        self.assertEqual(comparison.compare_mean(2, 10, "n_clusters"), "no_direction")
        self.assertEqual(comparison.compare_mean(.5, .5, "acc"), "tie")
        self.assertEqual(comparison.compare_mean(.8, np.nan, "acc"), "no_reference")

    def test_partial_counts_percent_sample_sd_and_latest_success(self):
        self.save("cora", "densgnn", 1, .5, runtime=10)
        self.save("cora", "ucode", 1, .7)
        self.save("cora", "ucode", 1, .8, run="02")
        self.save("cora", "ucode", 1, .99, run="03", status="failed", error="OOT")
        self.save("cora", "ucode", 2, .6)
        self.save("cora", "ns4gc", 1, .4, runtime=5)
        self.save("wiki", "ucode", 1, .9)
        report = self.report()
        self.assertEqual(report.table.loc[("UCoDe", "ACC ↑"), "Cora"], "70.00 ± 14.14")
        self.assertEqual(report.table.loc[("DensGNN (adaptive)", "ACC ↑"), "Cora"], "50.00 (n=1/2)")
        self.assertEqual(report.relations.loc[("UCoDe", "ACC ↑"), "Cora"], "better")
        self.assertEqual(report.relations.loc[("NS4GC", "ACC ↑"), "Cora"], "worse")
        self.assertEqual(report.relations.loc[("NS4GC", "Tempo (s) ↓"), "Cora"], "better")
        self.assertEqual(report.relations.loc[("UCoDe", "ACC ↑"), "Wiki"], "no_reference")
        self.assertEqual(report.table.index[-1][0], "DensGNN (adaptive)")
        self.assertEqual(len(report.attempts), 7)
        self.assertEqual(len(report.selected), 5)

    def test_no_results_is_pending_without_crashing(self):
        report = self.report()
        self.assertTrue(report.attempts.empty)
        self.assertTrue(report.table.map(lambda text: text == "Pendente (0/2)").all().all())
        self.assertTrue(report.audit.n_pending.eq(2).all())
        self.assertEqual(len(report.audit), 6)
        comparison.export_report(report, self.root / "exports")

    def test_oot_and_undefined_are_not_zero_scores(self):
        self.save("cora", "ucode", 1, .99, status="failed", error="OOT (out of time)")
        self.save("cora", "ucode", 2, .99, status="failed", error="OOT (out of time)")
        self.save("cora", "densgnn", 1, np.nan)
        report = self.report(("acc", "nmi"))
        self.assertEqual(report.table.loc[("UCoDe", "ACC ↑"), "Cora"], "OOT (0/2)")
        self.assertEqual(report.table.loc[("DensGNN (adaptive)", "ACC ↑"), "Cora"], "Indefinido (0/2)")
        self.assertEqual(report.table.loc[("UCoDe", "NMI ↑"), "Cora"], "N/A")
        summary = report.summary.query("model == 'ucode' and dataset == 'cora_benchmark_adaptive' and metric == 'acc'").iloc[0]
        self.assertTrue(np.isnan(summary["mean"]))
        self.assertEqual(summary.n_oot, 2)
        self.assertEqual(summary.n_pending, 0)

    def test_partial_csv_and_unconfigured_seed_are_reported(self):
        path = self.save("cora", "ucode", 1, .8)
        path.write_text("model,seed,status\nucode,1,\n")
        self.save("wiki", "ucode", 999, .9)
        report = self.report()
        self.assertEqual(len(report.issues), 2)
        self.assertTrue(report.selected.empty)
        self.assertTrue(report.audit.n_pending.eq(2).all())

    def test_result_paths_outside_current_results_are_rejected(self):
        path = self.config_dir / "core_border_adaptive_cora/ucode.yaml"
        config = yaml.safe_load(path.read_text())
        config["experiment"]["output_root"] = "other_outputs"
        path.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "somente results/"):
            self.report()

    def test_exports_preserve_colors_without_bolding_maxima(self):
        self.save("cora", "densgnn", 1, .5)
        self.save("cora", "ucode", 1, .8)
        self.save("cora", "ns4gc", 1, .4)
        report = self.report()
        output = self.root / "exports"
        comparison.export_report(report, output)
        html = (output / "comparison_vs_densgnn.html").read_text()
        latex = (output / "comparison_vs_densgnn.tex").read_text()
        self.assertIn("color: #b91c1c", html)
        self.assertIn("color: #1d4ed8", html)
        self.assertIn(r"\textcolor[HTML]{B91C1C}", latex)
        self.assertIn(r"\textcolor[HTML]{1D4ED8}", latex)
        self.assertNotIn(r"\textbf", latex)
        self.assertIn("summary_numeric.csv", [path.name for path in output.iterdir()])

    def test_core_border_use_separate_sources_and_same_dataset_column(self):
        folder = self.config_dir / "core_border_adaptive_cora"
        for variant in ("core", "border"):
            config = yaml.safe_load((folder / "densgnn.yaml").read_text())
            config["experiment"]["dataset"] = f"cora_{variant}"
            config["models"]["densgnn"]["params"]["point_selection"] = variant
            (folder / f"{variant}.yaml").write_text(yaml.safe_dump(config))
        # The old adaptive file must not create a duplicate reference.
        (folder / "adaptive.yaml").write_text((folder / "densgnn.yaml").read_text())
        self.save("cora", "densgnn", 1, .5)
        self.save("cora", "densgnn", 1, .7, source_dataset="cora_core")
        self.save("cora", "densgnn", 1, .4, source_dataset="cora_border")
        report = self.report(("acc",))
        self.assertEqual(list(report.table.columns), ["Cora", "Wiki"])
        self.assertEqual(report.table.loc[("DensGNN (core)", "ACC ↑"), "Cora"], "70.00 (n=1/2)")
        self.assertEqual(report.relations.loc[("DensGNN (core)", "ACC ↑"), "Cora"], "better")
        self.assertEqual(report.relations.loc[("DensGNN (border)", "ACC ↑"), "Cora"], "worse")
        self.assertEqual(report.table.loc[("DensGNN (core)", "ACC ↑"), "Wiki"], "N/A")
        self.assertEqual(list(report.table.index.get_level_values(0)[-3:]),
                         ["DensGNN (core)", "DensGNN (border)", "DensGNN (adaptive)"])
        self.assertEqual(len(report.selected), 3)
        self.assertEqual(set(report.selected.source_dataset), {"cora_core", "cora_border", "cora_benchmark_adaptive"})

    def test_internal_table_uses_same_snapshot_units_and_correct_direction(self):
        for path in self.config_dir.glob("*/*.yaml"):
            config = yaml.safe_load(path.read_text())
            config["metrics"]["include"] += list(comparison.INTERNAL_TABLE_METRICS)
            path.write_text(yaml.safe_dump(config))
        self.save("cora", "densgnn", 1, .5, silhouette=.5, calinski_harabasz=50,
                  davies_bouldin=.6, modularity=.5)
        self.save("cora", "ucode", 1, .4, silhouette=.8, calinski_harabasz=40,
                  davies_bouldin=.3, modularity=.4)
        snapshot = self.report(["acc", *comparison.INTERNAL_TABLE_METRICS])
        # New disk results must not change either table derived from the snapshot.
        self.save("cora", "densgnn", 2, .9)
        external = snapshot.for_metrics(["acc"])
        internal = snapshot.for_metrics(comparison.INTERNAL_TABLE_METRICS, percent=False)
        self.assertEqual(external.snapshot, internal.snapshot)
        self.assertTrue(external.audit.equals(internal.audit))
        self.assertEqual(internal.table.loc[("UCoDe", "Silhouette ↑"), "Cora"], "0.8000 (n=1/2)")
        self.assertEqual(internal.relations.loc[("UCoDe", "DB ↓"), "Cora"], "better")
        self.assertEqual(internal.relations.loc[("UCoDe", "CH ↑"), "Cora"], "worse")
        self.assertEqual(internal.relations.loc[("UCoDe", "Modularidade ↑"), "Cora"], "worse")
        self.assertEqual(external.table.loc[("DensGNN (adaptive)", "ACC ↑"), "Cora"], "50.00 (n=1/2)")
        output = self.root / "internal_exports"
        comparison.export_report(internal, output, table_label="tab:benchmark-internal")
        self.assertIn(r"\label{tab:benchmark-internal}", (output / "comparison_vs_densgnn.tex").read_text())

    def test_largest_subset_matches_exhaustive_optimum(self):
        rng = np.random.default_rng(42)
        for _ in range(12):
            better = rng.integers(0, 2, size=(8, 5))
            selected = comparison._largest_feasible_subset(better, 2)
            self.assertTrue((better[selected].sum(axis=0) <= 2).all())
            optimum = max(size for size in range(9) if any(
                (better[list(indices)].sum(axis=0) <= 2).all()
                for indices in combinations(range(8), size)
            ))
            self.assertEqual(len(selected), optimum)

    def test_top_three_selection_preserves_data_and_top_two_is_stricter(self):
        for dataset in ("cora", "wiki"):
            self.save(dataset, "densgnn", 1, .5, runtime=10)
            self.save(dataset, "ns4gc", 1, .7, runtime=5)
            self.save(dataset, "ucode", 1, .8, runtime=4)
        report = self.report(("acc", "runtime_seconds"))
        original = report.table.copy(deep=True)
        top_three = comparison.select_top_rank_subset(report, ["cora", "wiki"], ["acc", "runtime_seconds"], 3)
        top_two = comparison.select_top_rank_subset(report, ["cora", "wiki"], ["acc", "runtime_seconds"], 2)
        self.assertEqual(top_three.selected_models, ["ns4gc", "ucode", "densgnn"])
        self.assertTrue(top_three.ranks.rank_selected.eq(3).all())
        self.assertEqual(top_two.selected_models, ["ns4gc", "densgnn"])
        self.assertTrue(top_two.ranks.rank_selected.eq(2).all())
        self.assertTrue(top_two.ranks.rank_all_available.eq(3).all())
        self.assertTrue(report.table.equals(original))
        self.assertTrue(top_three.report.summary["mean"].equals(report.summary["mean"]))

    def test_selection_waits_for_reference_and_excludes_missing_competitors(self):
        self.save("cora", "densgnn", 1, .5)
        report = self.report(("acc",))
        pending = comparison.select_top_rank_subset(report, ["cora", "wiki"], ["acc"])
        self.assertIsNone(pending.report)
        self.assertEqual(pending.pending_reference.dataset.tolist(), ["wiki_benchmark_adaptive"])
        self.save("wiki", "densgnn", 1, .5)
        self.save("cora", "ucode", 1, .9)
        for dataset in ("cora", "wiki"):
            self.save(dataset, "ns4gc", 1, .4)
        ready = comparison.select_top_rank_subset(self.report(("acc",)), ["cora", "wiki"], ["acc"])
        self.assertEqual(ready.selected_models, ["ns4gc", "densgnn"])
        self.assertIn("Dados insuficientes", ready.exclusions.reason.tolist())

    def test_selection_ties_do_not_count_as_better(self):
        for dataset in ("cora", "wiki"):
            for model in ("densgnn", "ns4gc", "ucode"):
                self.save(dataset, model, 1, .5)
        selection = comparison.select_top_rank_subset(self.report(("acc",)), ["cora", "wiki"], ["acc"], 1)
        self.assertEqual(len(selection.selected_models), 3)
        self.assertTrue(selection.ranks.rank_selected.eq(1).all())

    def test_grouped_constraint_allows_losses_in_other_metrics(self):
        for dataset in ("cora", "wiki"):
            self.save(dataset, "densgnn", 1, .5, runtime=10)
            self.save(dataset, "ns4gc", 1, .7, runtime=15)
            self.save(dataset, "ucode", 1, .8, runtime=20)
        report = self.report(("acc", "runtime_seconds"))
        selection = comparison.select_top_rank_subset(
            report, ["cora", "wiki"], ["acc", "runtime_seconds"], max_rank=1,
            metric_groups={"example_group": ["acc", "runtime_seconds"]},
            min_metrics_per_group=1,
        )
        self.assertEqual(len(selection.selected_models), 3)
        self.assertTrue(selection.ranks.loc[selection.ranks.metric.eq("acc"), "rank_selected"].eq(3).all())
        self.assertTrue(selection.ranks.loc[selection.ranks.metric.eq("runtime_seconds"), "rank_selected"].eq(1).all())
        self.assertTrue(selection.coverage.n_within_limit.eq(1).all())
        self.assertIn("pelo menos 1", selection.report.selection_note)

    def test_grouped_optimizer_matches_exhaustive_optimum(self):
        rng = np.random.default_rng(17)
        groups = [[0, 1], [2, 3, 4]]
        for _ in range(12):
            better = rng.integers(0, 2, size=(8, 5))

            def feasible(indices):
                passes = better[list(indices)].sum(axis=0) <= 1
                return all(np.count_nonzero(passes[group]) >= 1 for group in groups)

            selected = comparison._largest_feasible_subset(better, 1, groups, 1)
            optimum = max(size for size in range(9) if any(
                feasible(indices) for indices in combinations(range(8), size)
            ))
            self.assertTrue(feasible(selected))
            self.assertEqual(len(selected), optimum)

    def test_ranked_latex_has_no_colors_and_marks_largest_and_second(self):
        self.save("cora", "densgnn", 1, .5, runtime=10)
        self.save("cora", "ns4gc", 1, .8, runtime=20)
        self.save("cora", "ucode", 1, .7, runtime=5)
        output = self.root / "ranked"
        comparison.export_report(self.report(("acc", "runtime_seconds")), output, latex_style="ranked")
        latex = (output / "comparison_vs_densgnn.tex").read_text()
        self.assertNotIn(r"\textcolor", latex)
        self.assertNotIn("xcolor", latex)
        self.assertIn(r"\textbf{80.00 (n=1/2)}", latex)
        self.assertIn(r"\underline{70.00 (n=1/2)}", latex)
        # The user requested numeric largest/second largest, not best direction.
        self.assertIn(r"\textbf{20.00 (n=1/2)}", latex)
        self.assertIn(r"\underline{10.00 (n=1/2)}", latex)

    def test_ranked_latex_shares_highlight_for_ties(self):
        self.save("cora", "densgnn", 1, .8)
        self.save("cora", "ns4gc", 1, .8)
        self.save("cora", "ucode", 1, .7)
        output = self.root / "tied_ranked"
        comparison.export_report(self.report(("acc",)), output, latex_style="ranked")
        latex = (output / "comparison_vs_densgnn.tex").read_text()
        self.assertEqual(latex.count(r"\textbf{80.00 (n=1/2)}"), 2)
        self.assertEqual(latex.count(r"\underline{70.00 (n=1/2)}"), 1)


if __name__ == "__main__":
    unittest.main()

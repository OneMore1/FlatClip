from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


VARIANTS = [
    ("real_flatmap", "Real flatmap"),
    ("left_right_swap", "Left-right hemisphere swap"),
    ("within_hemi_region_shuffle", "Within-hemi region shuffle"),
    ("phase_matched_randomization", "Phase/frequency-matched randomization"),
    ("spatial_block_shuffle", "Spatial block-shuffle"),
    ("random_vertex_permutation", "Vertex-permutation"),
    ("random_initialize", "Random initialize"),
    ("fc_heatmap", "FC heatmap"),
    ("random_image", "Random image"),
]

METRICS = [
    ("acc", "ACC"),
    ("weighted_f1", "wF1"),
    ("balanced_acc", "Balanced ACC"),
    ("macro_f1", "Macro F1"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize SigLIP2 HCP geometry-control results.")
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--feature-mode", default="cls_mean_token")
    p.add_argument("--out-prefix", default="geometry_control_siglip2_cls_mean_token")
    return p.parse_args()


def load_per_run(results_root: Path, variant: str, feature_mode: str) -> pd.DataFrame:
    path = results_root / variant / f"per_run_metrics_{feature_mode}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df.sort_values("seed").reset_index(drop=True)


def star(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} $\\pm$ {std * 100:.2f}"


def holm_correct(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvals[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    per_run = {variant: load_per_run(args.results_root, variant, args.feature_mode) for variant, _ in VARIANTS}

    summary_rows = []
    for variant, label in VARIANTS:
        row = {"variant": variant, "control_input": label}
        df = per_run[variant]
        for metric, _ in METRICS:
            vals = df[f"test_{metric}"].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1))
            row[f"{metric}_values"] = vals.tolist()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    stat_rows = []
    real = per_run["real_flatmap"].sort_values("seed")
    for metric, label in METRICS:
        raw = []
        controls = []
        for variant, control_label in VARIANTS[1:]:
            cur = per_run[variant].sort_values("seed")
            if real["seed"].tolist() != cur["seed"].tolist():
                raise ValueError(f"Seed mismatch for {variant}")
            diff = real[f"test_{metric}"].to_numpy(float) - cur[f"test_{metric}"].to_numpy(float)
            test = stats.ttest_rel(
                real[f"test_{metric}"].to_numpy(float),
                cur[f"test_{metric}"].to_numpy(float),
                alternative="greater",
            )
            raw.append(float(test.pvalue))
            controls.append((variant, control_label, diff))
        adj = holm_correct(raw)
        for (variant, control_label, diff), p, p_holm in zip(controls, raw, adj):
            stat_rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "comparison": f"Real flatmap > {control_label}",
                    "control_variant": variant,
                    "delta_mean": float(diff.mean()),
                    "delta_std": float(diff.std(ddof=1)),
                    "p_raw": p,
                    "p_holm": float(p_holm),
                    "stars": star(float(p_holm)),
                }
            )
    stats_df = pd.DataFrame(stat_rows)

    args.results_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_root / f"{args.out_prefix}_summary.csv"
    stats_path = args.results_root / f"{args.out_prefix}_paired_stats.csv"
    json_path = args.results_root / f"{args.out_prefix}_summary.json"
    tex_path = args.results_root / f"{args.out_prefix}_table.tex"
    summary.to_csv(summary_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    json_path.write_text(
        json.dumps({"summary": summary_rows, "paired_stats": stat_rows}, indent=2),
        encoding="utf-8",
    )

    stat_lookup = {
        (row.control_variant, row.metric): row.stars
        for row in stats_df.itertuples(index=False)
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\caption{Geometry-control ablation on HCP sex classification.",
        r"All results use the frozen SigLIP2 encoder with one global token per frame, followed by the same MLP probe.",
        r"Results are reported as mean$\pm$std over three seeds.",
        r"Significance markers indicate paired one-sided $t$-tests comparing Real flatmap against each control after Holm correction.",
        r"\best{Red} indicates the best performance.}",
        r"\label{tab:hcp_geometry_control_siglip2}",
        "",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"\textbf{Control input} &",
        r"\textbf{ACC $\uparrow$} &",
        r"\textbf{wF1 $\uparrow$} &",
        r"\textbf{Balanced ACC $\uparrow$} &",
        r"\textbf{Macro F1 $\uparrow$} \\",
        r"\midrule",
    ]
    for idx, row in summary.iterrows():
        variant = row["variant"]
        label = row["control_input"]
        cells = []
        for metric, _ in METRICS:
            text = fmt(row[f"{metric}_mean"], row[f"{metric}_std"])
            if variant == "real_flatmap":
                text = r"\best{" + text + "}"
            else:
                text = text + stat_lookup.get((variant, metric), "")
            cells.append(text)
        if variant == "real_flatmap":
            lines.append(r"\rowcolor{rowgray}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        lines.append("")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            "",
            r"\vspace{2pt}",
            r"\footnotesize{Markers denote Holm-corrected paired one-sided $t$-tests against Real flatmap: *$p<0.05$, **$p<0.01$.}",
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(summary[["control_input", "acc_mean", "weighted_f1_mean", "balanced_acc_mean", "macro_f1_mean"]])
    print(f"saved {summary_path}")
    print(f"saved {stats_path}")
    print(f"saved {tex_path}")


if __name__ == "__main__":
    main()

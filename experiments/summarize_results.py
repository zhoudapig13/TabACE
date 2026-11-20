from __future__ import annotations

import csv
from pathlib import Path


RESULTS_DIR = Path("results")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_float(value: str, digits: int = 4) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No results found._"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if key in {"metric_value", "runtime_sec", "rss_delta_mb", "mae"} and value:
                value = fmt_float(value, 4 if key != "runtime_sec" else 2)
            cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def main() -> None:
    single = read_csv(RESULTS_DIR / "single_dataset_results.csv")
    scale = read_csv(RESULTS_DIR / "scale_study_results.csv")
    benchmark = read_csv(RESULTS_DIR / "benchmark_results.csv")
    large_scale = read_csv(RESULTS_DIR / "large_scale_results.csv")
    finetune = read_csv(RESULTS_DIR / "finetune_results.csv")
    tiny_checkpoint = read_csv(RESULTS_DIR / "tiny_checkpoint_benchmark.csv")
    small_checkpoint = read_csv(RESULTS_DIR / "small_checkpoint_benchmark.csv")
    stage_mid_checkpoint = read_csv(RESULTS_DIR / "stage_mid_checkpoint_benchmark.csv")

    lines = [
        "# TabICL Reproducible Experiment Summary",
        "",
        "## 1. Minimal Reproducible Experiment",
        "",
        markdown_table(
            single,
            [
                ("Dataset", "dataset"),
                ("Task", "task"),
                ("Samples", "n_samples"),
                ("Features", "n_features"),
                ("Metric", "metric_name"),
                ("Score", "metric_value"),
                ("Runtime s", "runtime_sec"),
                ("Device", "device"),
            ],
        ),
        "",
        "## 2. Scale Study",
        "",
        markdown_table(
            scale,
            [
                ("Samples", "n_samples"),
                ("Features", "n_features"),
                ("Metric", "metric_name"),
                ("Score", "metric_value"),
                ("Runtime s", "runtime_sec"),
                ("RSS Delta MB", "rss_delta_mb"),
            ],
        ),
        "",
        "## 3. Multi-Dataset Benchmark",
        "",
        markdown_table(
            benchmark,
            [
                ("Dataset", "dataset"),
                ("Task", "task"),
                ("Samples", "n_samples"),
                ("Features", "n_features"),
                ("Metric", "metric_name"),
                ("Score", "metric_value"),
                ("MAE", "mae"),
                ("Runtime s", "runtime_sec"),
            ],
        ),
        "",
        "## 4. Larger Scale Study",
        "",
        markdown_table(
            large_scale,
            [
                ("Samples", "n_samples"),
                ("Features", "n_features"),
                ("Metric", "metric_name"),
                ("Score", "metric_value"),
                ("Runtime s", "runtime_sec"),
                ("RSS Delta MB", "rss_delta_mb"),
            ],
        ),
        "",
        "## 5. Downstream Fine-Tuning",
        "",
        markdown_table(
            finetune,
            [
                ("Stage", "stage"),
                ("Dataset", "dataset"),
                ("Task", "task"),
                ("Samples", "n_samples"),
                ("Metric", "metric_name"),
                ("Score", "metric_value"),
                ("MAE", "mae"),
                ("Runtime s", "runtime_sec"),
                ("Checkpoint", "checkpoint"),
            ],
        ),
        "",
        "## 6. Tiny From-Scratch Checkpoint Benchmark",
        "",
        markdown_table(
            tiny_checkpoint,
            [
                ("Dataset", "dataset"),
                ("Task", "task"),
                ("Samples", "n_samples"),
                ("Features", "n_features"),
                ("Checkpoint", "model_path"),
                ("Metric", "metric_name"),
                ("Score", "metric_value"),
                ("Runtime s", "runtime_sec"),
            ],
        ),
        "",
        "## 7. Small From-Scratch Checkpoint Benchmark",
        "",
        markdown_table(
            small_checkpoint,
            [
                ("Dataset", "dataset"),
                ("Task", "task"),
                ("Samples", "n_samples"),
                ("Features", "n_features"),
                ("Checkpoint", "model_path"),
                ("Metric", "metric_name"),
                ("Score", "metric_value"),
                ("Runtime s", "runtime_sec"),
            ],
        ),
        "",
        "## 8. Stage-Like Warm-Up Checkpoint Benchmark",
        "",
        markdown_table(
            stage_mid_checkpoint,
            [
                ("Dataset", "dataset"),
                ("Task", "task"),
                ("Samples", "n_samples"),
                ("Features", "n_features"),
                ("Checkpoint", "model_path"),
                ("Metric", "metric_name"),
                ("Score", "metric_value"),
                ("Runtime s", "runtime_sec"),
            ],
        ),
        "",
        "## Notes",
        "",
        "- All runs use TabICL only; no algorithm changes or baseline comparisons are included yet.",
        "- Defaults are intentionally conservative: one ensemble member and a fixed random seed.",
        "- Runtime is wall-clock time for fit plus predict on one train/test split.",
        "- The fine-tuning section adapts the pretrained TabICL checkpoint on a downstream dataset; it is not from-scratch foundation-model pretraining.",
        "- The tiny checkpoint section uses a newly trained 100-step toy checkpoint, so low accuracy is expected.",
        "- The small checkpoint section uses a 1000-step from-scratch checkpoint; it is still far below official-scale pretraining.",
        "- The stage-like warm-up uses a larger architecture but only 200 steps, so it validates scale-up mechanics rather than quality.",
    ]

    output = RESULTS_DIR / "summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

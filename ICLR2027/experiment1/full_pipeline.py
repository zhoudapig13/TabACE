from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = EXPERIMENT_DIR / "runs"


@dataclass(frozen=True)
class StageConfig:
    name: str
    num_batches: int
    max_steps: int
    batch_size: int
    micro_batch_size: int
    prior_type: str
    min_features: int
    max_features: int
    max_classes: int
    min_seq_len: int | None
    max_seq_len: int
    log_seq_len: bool
    seq_len_per_gp: bool
    replay_small: bool
    min_train_size: str
    max_train_size: str
    batch_size_per_gp: int
    lr: str
    scheduler: str
    warmup_proportion: str | None = None
    poly_decay_lr_end: str | None = None
    poly_decay_power: str | None = None
    freeze_col: bool = False
    freeze_row: bool = False
    save_temp_every: int = 50
    save_perm_every: int = 5000
    embed_dim: int = 128
    col_num_blocks: int = 3
    col_nhead: int = 4
    col_num_inds: int = 128
    row_num_blocks: int = 3
    row_nhead: int = 8
    row_num_cls: int = 4
    icl_num_blocks: int = 12
    icl_nhead: int = 4


SMOKE = [
    StageConfig(
        name="smoke_stage1",
        num_batches=2,
        max_steps=2,
        batch_size=4,
        micro_batch_size=1,
        prior_type="mlp_scm",
        min_features=2,
        max_features=4,
        max_classes=3,
        min_seq_len=None,
        max_seq_len=64,
        log_seq_len=False,
        seq_len_per_gp=False,
        replay_small=False,
        min_train_size="0.5",
        max_train_size="0.8",
        batch_size_per_gp=1,
        lr="1e-4",
        scheduler="constant",
        save_temp_every=1,
        save_perm_every=2,
        embed_dim=32,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=16,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
    )
]

TINY = [
    StageConfig(
        name="tiny_stage1",
        num_batches=100,
        max_steps=100,
        batch_size=8,
        micro_batch_size=1,
        prior_type="mlp_scm",
        min_features=2,
        max_features=30,
        max_classes=5,
        min_seq_len=None,
        max_seq_len=256,
        log_seq_len=False,
        seq_len_per_gp=False,
        replay_small=False,
        min_train_size="0.5",
        max_train_size="0.9",
        batch_size_per_gp=1,
        lr="1e-4",
        scheduler="cosine_warmup",
        warmup_proportion="0.05",
        save_temp_every=25,
        save_perm_every=100,
        embed_dim=64,
        col_num_blocks=2,
        col_nhead=4,
        col_num_inds=32,
        row_num_blocks=2,
        row_nhead=4,
        row_num_cls=2,
        icl_num_blocks=2,
        icl_nhead=4,
    )
]

SMALL = [
    StageConfig(
        name="small_stage1",
        num_batches=1000,
        max_steps=1000,
        batch_size=32,
        micro_batch_size=2,
        prior_type="mix_scm",
        min_features=2,
        max_features=50,
        max_classes=10,
        min_seq_len=None,
        max_seq_len=512,
        log_seq_len=False,
        seq_len_per_gp=False,
        replay_small=False,
        min_train_size="0.5",
        max_train_size="0.9",
        batch_size_per_gp=2,
        lr="1e-4",
        scheduler="cosine_warmup",
        warmup_proportion="0.05",
        save_temp_every=100,
        save_perm_every=500,
        embed_dim=96,
        col_num_blocks=2,
        col_nhead=4,
        col_num_inds=64,
        row_num_blocks=2,
        row_nhead=4,
        row_num_cls=4,
        icl_num_blocks=4,
        icl_nhead=4,
    )
]

STAGE_LIKE = [
    StageConfig(
        name="stage_like_stage1",
        num_batches=5000,
        max_steps=5000,
        batch_size=128,
        micro_batch_size=4,
        prior_type="mix_scm",
        min_features=2,
        max_features=100,
        max_classes=10,
        min_seq_len=None,
        max_seq_len=1024,
        log_seq_len=False,
        seq_len_per_gp=False,
        replay_small=False,
        min_train_size="0.1",
        max_train_size="0.9",
        batch_size_per_gp=4,
        lr="1e-4",
        scheduler="cosine_warmup",
        warmup_proportion="0.02",
        save_temp_every=250,
        save_perm_every=1000,
    )
]

OFFICIAL = [
    StageConfig(
        name="official_stage1",
        num_batches=100000,
        max_steps=100000,
        batch_size=512,
        micro_batch_size=4,
        prior_type="mix_scm",
        min_features=2,
        max_features=100,
        max_classes=10,
        min_seq_len=None,
        max_seq_len=1024,
        log_seq_len=False,
        seq_len_per_gp=False,
        replay_small=False,
        min_train_size="0.1",
        max_train_size="0.9",
        batch_size_per_gp=4,
        lr="1e-4",
        scheduler="cosine_warmup",
        warmup_proportion="0.02",
        save_temp_every=50,
        save_perm_every=5000,
    ),
    StageConfig(
        name="official_stage2",
        num_batches=2000,
        max_steps=2000,
        batch_size=512,
        micro_batch_size=1,
        prior_type="mix_scm",
        min_features=2,
        max_features=100,
        max_classes=10,
        min_seq_len=1000,
        max_seq_len=40000,
        log_seq_len=True,
        seq_len_per_gp=True,
        replay_small=False,
        min_train_size="0.5",
        max_train_size="0.9",
        batch_size_per_gp=2,
        lr="2e-5",
        scheduler="polynomial_decay_warmup",
        warmup_proportion="0",
        poly_decay_lr_end="5e-6",
        poly_decay_power="2.0",
        save_temp_every=5,
        save_perm_every=100,
    ),
    StageConfig(
        name="official_stage3",
        num_batches=50,
        max_steps=50,
        batch_size=512,
        micro_batch_size=1,
        prior_type="mix_scm",
        min_features=2,
        max_features=100,
        max_classes=10,
        min_seq_len=40000,
        max_seq_len=60000,
        log_seq_len=True,
        seq_len_per_gp=True,
        replay_small=True,
        min_train_size="0.5",
        max_train_size="0.9",
        batch_size_per_gp=1,
        lr="2e-6",
        scheduler="constant",
        freeze_col=True,
        freeze_row=True,
        save_temp_every=1,
        save_perm_every=5,
    ),
]

PRESETS = {
    "smoke": SMOKE,
    "tiny": TINY,
    "small": SMALL,
    "stage_like": STAGE_LIKE,
    "official": OFFICIAL,
}


def bool_arg(value: bool) -> str:
    return "True" if value else "False"


def run(cmd: list[str], *, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n$ {printable}", flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(
        checkpoint_dir.glob("step-*.ckpt"),
        key=lambda path: int(path.stem.split("-")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def prior_command(python_bin: str, stage: StageConfig, prior_dir: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "tabicl.prior",
        "--save_dir",
        str(prior_dir),
        "--np_seed",
        str(args.np_seed),
        "--torch_seed",
        str(args.torch_seed),
        "--num_batches",
        str(stage.num_batches),
        "--resume_from",
        "0",
        "--batch_size",
        str(stage.batch_size),
        "--batch_size_per_gp",
        str(stage.batch_size_per_gp),
        "--min_features",
        str(stage.min_features),
        "--max_features",
        str(stage.max_features),
        "--max_classes",
        str(stage.max_classes),
        "--max_seq_len",
        str(stage.max_seq_len),
        "--min_train_size",
        stage.min_train_size,
        "--max_train_size",
        stage.max_train_size,
        "--prior_type",
        stage.prior_type,
        "--n_jobs",
        str(args.n_jobs),
        "--num_threads_per_generate",
        str(args.num_threads_per_generate),
        "--device",
        args.prior_device,
    ]
    if stage.min_seq_len is not None:
        cmd.extend(["--min_seq_len", str(stage.min_seq_len)])
    if stage.log_seq_len:
        cmd.extend(["--log_seq_len", "True"])
    if stage.seq_len_per_gp:
        cmd.extend(["--seq_len_per_gp", "True"])
    if stage.replay_small:
        cmd.extend(["--replay_small", "True"])
    return cmd


def train_command(
    python_bin: str,
    stage: StageConfig,
    prior_dir: Path | None,
    checkpoint_dir: Path,
    previous_checkpoint: Path | None,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "tabicl.train",
        "--wandb_log",
        bool_arg(args.wandb_log),
        "--wandb_project",
        args.wandb_project,
        "--wandb_name",
        stage.name,
        "--wandb_mode",
        args.wandb_mode,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--np_seed",
        str(args.np_seed),
        "--torch_seed",
        str(args.torch_seed),
        "--max_steps",
        str(stage.max_steps),
        "--batch_size",
        str(stage.batch_size),
        "--micro_batch_size",
        str(stage.micro_batch_size),
        "--lr",
        stage.lr,
        "--scheduler",
        stage.scheduler,
        "--gradient_clipping",
        "1.0",
        "--prior_device",
        args.prior_device,
        "--batch_size_per_gp",
        str(stage.batch_size_per_gp),
        "--min_features",
        str(stage.min_features),
        "--max_features",
        str(stage.max_features),
        "--max_classes",
        str(stage.max_classes),
        "--max_seq_len",
        str(stage.max_seq_len),
        "--min_train_size",
        stage.min_train_size,
        "--max_train_size",
        stage.max_train_size,
        "--prior_type",
        stage.prior_type,
        "--embed_dim",
        str(stage.embed_dim),
        "--col_num_blocks",
        str(stage.col_num_blocks),
        "--col_nhead",
        str(stage.col_nhead),
        "--col_num_inds",
        str(stage.col_num_inds),
        "--col_feature_group",
        "False",
        "--row_num_blocks",
        str(stage.row_num_blocks),
        "--row_nhead",
        str(stage.row_nhead),
        "--row_num_cls",
        str(stage.row_num_cls),
        "--row_rope_base",
        "100000",
        "--icl_num_blocks",
        str(stage.icl_num_blocks),
        "--icl_nhead",
        str(stage.icl_nhead),
        "--ff_factor",
        "2",
        "--norm_first",
        "True",
        "--freeze_col",
        bool_arg(stage.freeze_col),
        "--freeze_row",
        bool_arg(stage.freeze_row),
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--save_temp_every",
        str(stage.save_temp_every),
        "--save_perm_every",
        str(stage.save_perm_every),
        "--max_checkpoints",
        str(args.max_checkpoints),
    ]
    if stage.warmup_proportion is not None:
        cmd.extend(["--warmup_proportion", stage.warmup_proportion])
    if stage.poly_decay_lr_end is not None:
        cmd.extend(["--poly_decay_lr_end", stage.poly_decay_lr_end])
    if stage.poly_decay_power is not None:
        cmd.extend(["--poly_decay_power", stage.poly_decay_power])
    if prior_dir is not None:
        cmd.extend(
            [
                "--prior_dir",
                str(prior_dir),
                "--load_prior_start",
                "0",
                "--delete_after_load",
                bool_arg(args.delete_prior_after_load),
            ]
        )
    if stage.min_seq_len is not None:
        cmd.extend(["--min_seq_len", str(stage.min_seq_len)])
    if stage.log_seq_len:
        cmd.extend(["--log_seq_len", "True"])
    if stage.seq_len_per_gp:
        cmd.extend(["--seq_len_per_gp", "True"])
    if stage.replay_small:
        cmd.extend(["--replay_small", "True"])
    if previous_checkpoint is not None:
        cmd.extend(["--checkpoint_path", str(previous_checkpoint), "--only_load_model", "True"])
    return cmd


def eval_command(
    python_bin: str,
    checkpoint: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        python_bin,
        "experiment1/run_reproduction.py",
        "--datasets",
        args.eval_datasets,
        "--synthetic-samples",
        str(args.eval_synthetic_samples),
        "--n-estimators",
        str(args.eval_n_estimators),
        "--device",
        args.eval_device,
        "--random-state",
        str(args.np_seed),
        "--classifier-model-path",
        str(checkpoint),
        "--output",
        str(run_dir / "eval_reproduction_results.csv"),
    ]


def scale_eval_command(
    python_bin: str,
    checkpoint: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        python_bin,
        "experiment1/run_scale_reproduction.py",
        "--datasets",
        args.scale_datasets,
        "--scales",
        args.scale_values,
        "--n-estimators",
        str(args.eval_n_estimators),
        "--device",
        args.eval_device,
        "--random-state",
        str(args.np_seed),
        "--classifier-model-path",
        str(checkpoint),
        "--output",
        str(run_dir / "scale_reproduction_results.csv"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full official-style TabICL reproduction pipeline.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--prior-device", default="cpu")
    parser.add_argument("--np-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--num-threads-per-generate", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument(
        "--prior-mode",
        choices=["pregen", "on_the_fly"],
        default="pregen",
        help="Use pre-generated prior files or generate prior batches inside the trainer.",
    )
    parser.add_argument("--skip-prior", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-scale-eval", action="store_true")
    parser.add_argument("--delete-prior-after-load", action="store_true")
    parser.add_argument("--wandb-log", action="store_true")
    parser.add_argument("--wandb-project", default="TabICL")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--eval-datasets", default="breast_cancer,wine,synthetic_classification")
    parser.add_argument("--eval-synthetic-samples", type=int, default=1000)
    parser.add_argument("--eval-n-estimators", type=int, default=1)
    parser.add_argument("--scale-datasets", default="synthetic_classification")
    parser.add_argument("--scale-values", default="100,300,1000,3000")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stages = PRESETS[args.preset]
    run_name = args.run_name or args.preset
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "preset": args.preset,
        "run_name": run_name,
        "stages": [asdict(stage) for stage in stages],
        "args": vars(args),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    previous_checkpoint: Path | None = None
    for stage in stages:
        prior_dir = run_dir / f"{stage.name}_prior"
        checkpoint_dir = run_dir / f"{stage.name}_ckpts"
        if args.prior_mode == "pregen":
            prior_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if args.prior_mode == "pregen" and not args.skip_prior:
            run(prior_command(args.python_bin, stage, prior_dir, args), dry_run=args.dry_run)

        if not args.skip_train:
            run(
                train_command(
                    args.python_bin,
                    stage,
                    prior_dir if args.prior_mode == "pregen" else None,
                    checkpoint_dir,
                    previous_checkpoint,
                    args,
                ),
                dry_run=args.dry_run,
            )

        if args.dry_run:
            previous_checkpoint = checkpoint_dir / f"step-{stage.max_steps}.ckpt"
        else:
            previous_checkpoint = latest_checkpoint(checkpoint_dir)
            if previous_checkpoint is None and not args.skip_train:
                raise RuntimeError(f"No checkpoint was produced under {checkpoint_dir}")

    if not args.skip_eval:
        if previous_checkpoint is None:
            raise RuntimeError("Evaluation needs a checkpoint. Run training first or disable --skip-eval.")
        run(eval_command(args.python_bin, previous_checkpoint, run_dir, args), dry_run=args.dry_run)

    if not args.skip_scale_eval:
        if previous_checkpoint is None:
            raise RuntimeError("Scale evaluation needs a checkpoint. Run training first or disable --skip-scale-eval.")
        run(scale_eval_command(args.python_bin, previous_checkpoint, run_dir, args), dry_run=args.dry_run)

    print(f"\n[experiment1] Full pipeline manifest: {manifest_path}")
    if previous_checkpoint is not None:
        print(f"[experiment1] Last checkpoint: {previous_checkpoint}")


if __name__ == "__main__":
    main()

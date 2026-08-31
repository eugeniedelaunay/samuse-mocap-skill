"""
run_pipeline.py -- SAMuSe end-to-end pipeline orchestrator.

Runs every stage in dependency order, reading defaults from config.yaml so
no one has to edit source files or guess paths. Works identically on
Windows PowerShell, macOS, and Linux because it only shells out to
`sys.executable -m` calls -- no OS-specific path syntax.

Stages (in order):
  1. preprocess          raw MoCap CSVs           -> cleaned trial CSVs
  2. unified_pca         cleaned trial CSVs        -> unified PCA participant features
  3. extract_features    cleaned + PCA features    -> trial_features.csv
  4. merge_skill_labels  trial_features.csv + metadata -> trial_features_with_skill.csv
  5. analyze_features    trial_features_with_skill.csv -> QC/diagnostic reports
  6. train_classical     trial_features_with_skill.csv -> trial-wise classical results
  7. train_grouped       trial_features_with_skill.csv -> participant-wise classical results
  8. train_rnn           cleaned trial CSVs         -> RNN results (both protocols)

Any stage failure stops the pipeline immediately with a clear, actionable
error message naming the failing stage, the command that was run, and the
captured stderr tail.

Usage
-----
python run_pipeline.py --config config.yaml
python run_pipeline.py --config config.yaml --skip-rnn
python run_pipeline.py --config config.yaml --only preprocess,extract_features
python run_pipeline.py --config config.yaml --dry-run
"""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(
        "ERROR: PyYAML is not installed. Run:\n"
        "  pip install -r requirements.txt\n"
        "before running the pipeline."
    )

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

ALL_STAGES = [
    "preprocess",
    "unified_pca",
    "extract_features",
    "merge_skill_labels",
    "analyze_features",
    "train_classical",
    "train_grouped",
    "train_rnn",
]


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        sys.exit(
            f"ERROR: config file not found: {config_path}\n"
            f"Copy config.yaml from the repo root or pass --config <path>."
        )
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    required_top_level = ["paths", "preprocessing", "feature_extraction", "cross_validation", "rnn"]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        sys.exit(f"ERROR: config.yaml is missing required section(s): {missing}")
    return config


def resolve_path(relative: str) -> Path:
    return (REPO_ROOT / relative).resolve()


def run_command(command: list, stage_name: str, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(str(part)) for part in command)
    print(f"\n{'=' * 90}\n[STAGE] {stage_name}\n{'=' * 90}\n{printable}\n")
    if dry_run:
        print(f"[DRY RUN] Skipped execution of stage: {stage_name}")
        return

    started = time.time()
    result = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
    elapsed = time.time() - started

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-25:]) if result.stderr else "(no stderr captured)"
        sys.exit(
            f"\n{'!' * 90}\n"
            f"PIPELINE FAILED at stage: {stage_name}\n"
            f"Command: {printable}\n"
            f"Exit code: {result.returncode}\n"
            f"Last lines of stderr:\n{tail}\n"
            f"{'!' * 90}\n"
            f"Fix the error above and re-run. You can resume from this stage with:\n"
            f"  python run_pipeline.py --config {sys.argv[sys.argv.index('--config') + 1] if '--config' in sys.argv else 'config.yaml'} "
            f"--only {stage_name},{','.join(ALL_STAGES[ALL_STAGES.index(stage_name) + 1:])}"
        )
    print(f"[OK] {stage_name} finished in {elapsed:.1f}s")


def stage_preprocess(config: dict, dry_run: bool) -> None:
    paths = config["paths"]
    pre = config["preprocessing"]
    command = [
        sys.executable, str(SRC_DIR / "preprocess.py"),
        "--input_dir", str(resolve_path(paths["raw_mocap_dir"])),
        "--output_dir", str(resolve_path(paths["preprocessed_dir"])),
        "--cutoff", str(pre["default_cutoff_hz"]),
    ]
    run_command(command, "preprocess", dry_run)


def stage_unified_pca(config: dict, dry_run: bool) -> None:
    paths = config["paths"]
    pca_cfg = config["unified_pca"]
    command = [
        sys.executable, str(SRC_DIR / "unified_pca_evaluation.py"),
        "--input_dir", str(resolve_path(paths["preprocessed_dir"])),
        "--trial_features", str(resolve_path(paths["features_dir"]) / "trial_features.csv"),
        "--metadata_file", str(resolve_path(paths["metadata_file"])),
        "--output_dir", str(resolve_path(paths["unified_pca_eval_dir"])),
        "--condition", *pca_cfg["conditions"],
    ]
    run_command(command, "unified_pca", dry_run)


def stage_extract_features(config: dict, dry_run: bool) -> None:
    paths = config["paths"]
    fe = config["feature_extraction"]
    pca_csv = resolve_path(paths["unified_pca_eval_dir"]) / "unified_pca_participant_features.csv"
    command = [
        sys.executable, str(SRC_DIR / "extract_features3.py"),
        "--input_dir", str(resolve_path(paths["preprocessed_dir"])),
        "--output_dir", str(resolve_path(paths["features_dir"])),
        "--metadata_file", str(resolve_path(paths["metadata_file"])),
        "--conditions", fe["conditions"],
        "--pca_features_csv", str(pca_csv),
    ]
    if not fe.get("normalize", True):
        command.append("--no_normalize")
    run_command(command, "extract_features", dry_run)


def stage_merge_skill_labels(config: dict, dry_run: bool) -> None:
    paths = config["paths"]
    trial_features = resolve_path(paths["features_dir"]) / "trial_features.csv"
    output_file = resolve_path(paths["features_dir"]) / "trial_features_with_skill.csv"
    command = [
        sys.executable, str(SRC_DIR / "merge_skill_labels.py"),
        "--trial_features", str(trial_features),
        "--metadata_file", str(resolve_path(paths["metadata_file"])),
        "--output_file", str(output_file),
    ]
    run_command(command, "merge_skill_labels", dry_run)


def stage_analyze_features(config: dict, dry_run: bool) -> None:
    paths = config["paths"]
    trial_features = resolve_path(paths["features_dir"]) / "trial_features_with_skill.csv"
    command = [
        sys.executable, str(SRC_DIR / "analyze_features2.py"),
        "--trial_features", str(trial_features),
        "--output_dir", str(resolve_path(paths["analysis_dir"])),
        "--target_col", "Skill Level",
        "--metadata_file", str(resolve_path(paths["metadata_file"])),
        "--per_instrument",
    ]
    run_command(command, "analyze_features", dry_run)


def stage_train_classical(config: dict, dry_run: bool, task: str, instrument_mode: str, feature_config: str) -> None:
    paths = config["paths"]
    cv = config["cross_validation"]
    trial_features = resolve_path(paths["features_dir"]) / "trial_features_with_skill.csv"
    output_dir = resolve_path(paths["training_results_dir"])
    command = [
        sys.executable, str(SRC_DIR / "train_skill_classifiers.py"),
        "--trial_features", str(trial_features),
        "--output_dir", str(output_dir),
        "--task", task,
        "--instrument_mode", instrument_mode,
        "--feature_config", feature_config,
        "--model", "all",
        "--n_splits", str(cv["n_splits"]),
    ]
    run_command(command, f"train_classical[{task}/{instrument_mode}/{feature_config}]", dry_run)


def stage_train_grouped(config: dict, dry_run: bool, task: str, instrument_mode: str, feature_config: str) -> None:
    paths = config["paths"]
    cv = config["cross_validation"]
    trial_features = resolve_path(paths["features_dir"]) / "trial_features_with_skill.csv"
    output_dir = resolve_path(paths["training_results_grouped_dir"])
    command = [
        sys.executable, str(SRC_DIR / "train_skill_classifiers_grouped.py"),
        "--trial_features", str(trial_features),
        "--output_dir", str(output_dir),
        "--task", task,
        "--instrument_mode", instrument_mode,
        "--feature_config", feature_config,
        "--model", "all",
        "--n_splits", str(cv["n_splits"]),
    ]
    run_command(command, f"train_grouped[{task}/{instrument_mode}/{feature_config}]", dry_run)


def stage_train_rnn(config: dict, dry_run: bool, task: str, protocol: str) -> None:
    paths = config["paths"]
    rnn = config["rnn"]
    cv = config["cross_validation"]
    output_dir = resolve_path(paths["training_results_rnn_dir"]) / f"pooled_{task}_{protocol}_{rnn['rnn_type']}"
    command = [
        sys.executable, str(SRC_DIR / "train_rnn_skill.py"),
        "--input_dir", str(resolve_path(paths["preprocessed_dir"])),
        "--metadata_file", str(resolve_path(paths["metadata_file"])),
        "--output_dir", str(output_dir),
        "--instrument", "all",
        "--task", task,
        "--protocol", protocol,
        "--rnn_type", rnn["rnn_type"],
        "--hidden_dim", str(rnn["hidden_dim"]),
        "--num_layers", str(rnn["num_layers"]),
        "--dropout", str(rnn["dropout"]),
        "--batch_size", str(rnn["batch_size"]),
        "--epochs", str(rnn["epochs"]),
        "--patience", str(rnn["patience"]),
        "--learning_rate", str(rnn["learning_rate"]),
        "--weight_decay", str(rnn["weight_decay"]),
        "--n_splits", str(cv["n_splits"]),
        "--device", str(rnn.get("device", "auto")),
    ]
    run_command(command, f"train_rnn[{task}/{protocol}]", dry_run)


STAGE_DISPATCH = {
    "preprocess": lambda cfg, dry: stage_preprocess(cfg, dry),
    "unified_pca": lambda cfg, dry: stage_unified_pca(cfg, dry),
    "extract_features": lambda cfg, dry: stage_extract_features(cfg, dry),
    "merge_skill_labels": lambda cfg, dry: stage_merge_skill_labels(cfg, dry),
    "analyze_features": lambda cfg, dry: stage_analyze_features(cfg, dry),
    "train_classical": lambda cfg, dry: [
        stage_train_classical(cfg, dry, task, mode, feature_config)
        for task in ("3class", "2class")
        for mode in ("pooled", "separate")
        for feature_config in ("baseline", "shared_pca")
    ],
    "train_grouped": lambda cfg, dry: [
        stage_train_grouped(cfg, dry, task, mode, feature_config)
        for task in ("3class", "2class")
        for mode in ("pooled", "separate")
        for feature_config in ("baseline", "shared_pca")
    ],
    "train_rnn": lambda cfg, dry: [
        stage_train_rnn(cfg, dry, task, protocol)
        for task in ("3class", "2class")
        for protocol in ("trial", "participant")
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="SAMuSe one-command pipeline: preprocessing through classical + RNN training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--skip-rnn", action="store_true",
        help="Run only preprocessing + feature extraction + classical training (fast path, no PyTorch needed).",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help=f"Comma-separated subset of stages to run: {','.join(ALL_STAGES)}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print every command without executing it.")
    args = parser.parse_args()

    config_path = resolve_path(args.config) if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    if args.only:
        stages = [stage.strip() for stage in args.only.split(",") if stage.strip()]
        unknown = [stage for stage in stages if stage not in ALL_STAGES]
        if unknown:
            sys.exit(f"ERROR: unknown stage(s) in --only: {unknown}. Valid stages: {ALL_STAGES}")
    elif args.skip_rnn:
        stages = [stage for stage in ALL_STAGES if stage != "train_rnn"]
    else:
        stages = ALL_STAGES

    print(f"SAMuSe pipeline starting. Config: {config_path}")
    print(f"Stages to run: {stages}")
    if args.dry_run:
        print("DRY RUN MODE: no commands will actually execute.\n")

    pipeline_started = time.time()
    for stage in stages:
        STAGE_DISPATCH[stage](config, args.dry_run)
    total_elapsed = time.time() - pipeline_started

    print(f"\n{'=' * 90}\nPIPELINE COMPLETE. Total wall time: {total_elapsed / 60:.1f} minutes.\n{'=' * 90}")


if __name__ == "__main__":
    main()
"""
preprocess.py — SAMuSe MoCap preprocessing pipeline (Stages 1-3)

Stage 1: Loading & Cleaning
Stage 2: Gap Filling (linear interpolation)
Stage 3: Signal Filtering (Butterworth low-pass, zero-phase, per-joint-group cutoffs)
"""

import argparse
import re
import glob
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REFERENCE_N_COLUMNS = 199
METADATA_COLUMNS = {
    "participant_id", "condition", "block",
    "Timestamp", "Frame", "Sub Frame", "time_s",
}
GAP_LIMIT_FRAMES = 5
FILTER_ORDER = 4
SAMPLING_RATE_HZ = 50.0
DEFAULT_CUTOFF_HZ = 6.0  # fallback for unmatched joints

PLACEHOLDER_COL_PATTERN = re.compile(r"^\(.*\)$")

# Per-joint-group cutoff defaults, informed by biomechanics filtering literature.
# Distal/fast joints (wrist, hand, finger) need higher cutoffs than proximal/slow
# segments (trunk, shoulder, hip) — see residual analysis studies on gait and
# fast upper-limb tasks (typical ranges ~3-6 Hz proximal, ~8-12 Hz distal).
DEFAULT_JOINT_GROUP_CUTOFFS = {
    r"(?i)wrist|hand|finger|thumb":                              9.0,
    r"(?i)toe":                                                    8.0,
    r"(?i)elbow|forearm":                                          8.0,
    r"(?i)shoulder|clavicle|arm":                                  8.0,
    r"(?i)hip|pelvis|thigh|knee|ankle|foot":                       6.0,
    r"(?i)frontal plane (knee|projection) angle|\(deg\)\.1|\(ratio\)\.1": 6.0,
    r"(?i)spine|trunk|chest|neck|head|nose|ear|centerofgravity":    2.0,
}


# ---------------------------------------------------------------------------
# Stage 1: Loading & Cleaning
# ---------------------------------------------------------------------------
def load_and_clean(filepath: str) -> tuple[pd.DataFrame, dict]:
    log = {"file": os.path.basename(filepath), "warnings": []}

    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    if "Annotations" in df.columns:
        df = df.drop(columns=["Annotations"], errors="ignore")
        log["warnings"].append("dropped_annotations_column")

    placeholder_cols = [c for c in df.columns if PLACEHOLDER_COL_PATTERN.match(c)]
    if placeholder_cols:
        df = df.drop(columns=placeholder_cols)
        log["warnings"].append(f"dropped_placeholder_cols:{len(placeholder_cols)}")

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        log["warnings"].append(f"dropped_empty_cols:{len(empty_cols)}")

    if len(df.columns) != REFERENCE_N_COLUMNS:
        log["warnings"].append(
            f"column_count_mismatch:{len(df.columns)}_vs_{REFERENCE_N_COLUMNS}"
        )

    if "time_s" in df.columns:
        df = df.sort_values("time_s").reset_index(drop=True)
    else:
        log["warnings"].append("missing_time_s_column")

    log["n_columns"] = len(df.columns)
    log["n_rows"] = len(df)
    return df, log


# ---------------------------------------------------------------------------
# Stage 2: Gap Filling
# ---------------------------------------------------------------------------
def get_signal_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in METADATA_COLUMNS
        and ("Position" in c or "Angle" in c or "Angles" in c)
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def fill_gaps(df: pd.DataFrame, log: dict, limit: int = GAP_LIMIT_FRAMES) -> pd.DataFrame:
    signal_cols = get_signal_columns(df)
    n_nan_before = int(df[signal_cols].isna().sum().sum())

    if n_nan_before > 0:
        df[signal_cols] = df[signal_cols].interpolate(
            method="linear", limit=limit, limit_direction="both"
        )

    n_nan_after = int(df[signal_cols].isna().sum().sum())
    log["nan_before_interp"] = n_nan_before
    log["nan_after_interp"] = n_nan_after

    if n_nan_after > 0:
        log["warnings"].append(
            f"remaining_nans_after_interp:{n_nan_after} (gap>{limit} frames, needs manual review)"
        )

    return df


# ---------------------------------------------------------------------------
# Stage 3: Signal Filtering — per-joint-group cutoffs
# ---------------------------------------------------------------------------
def resolve_cutoff(
    col_name: str,
    joint_group_cutoffs: dict[str, float],
    default_cutoff: float = DEFAULT_CUTOFF_HZ,
) -> tuple[float, str | None]:
    """Match a column name against joint-group regex patterns.
    Returns (cutoff_hz, matched_pattern). Falls back to default_cutoff
    if no pattern matches. First match wins if multiple patterns overlap.
    """
    for pattern, cutoff in joint_group_cutoffs.items():
        if re.search(pattern, col_name):
            return cutoff, pattern
    return default_cutoff, None


def butterworth_filter(
    series: np.ndarray,
    cutoff: float,
    fs: float = SAMPLING_RATE_HZ,
    order: int = FILTER_ORDER,
) -> tuple[np.ndarray, bool]:
    """Zero-phase low-pass Butterworth filter.
    Returns (filtered_series, was_filtered). was_filtered=False if trial
    too short for filtfilt's padding requirement (caller must handle).
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    padlen = 3 * max(len(a), len(b))
    if len(series) <= padlen:
        return series, False

    return filtfilt(b, a, series), True


def apply_filtering(
    df: pd.DataFrame,
    log: dict,
    joint_group_cutoffs: dict[str, float] | None = None,
    default_cutoff: float = DEFAULT_CUTOFF_HZ,
) -> pd.DataFrame:
    """Apply Butterworth filter per column, using joint-group-specific cutoffs.
    Must run AFTER gap-filling. Logs which cutoff/pattern was used per column
    and any columns that fell back to the default.
    """
    joint_group_cutoffs = joint_group_cutoffs or DEFAULT_JOINT_GROUP_CUTOFFS
    signal_cols = get_signal_columns(df)

    cutoffs_used = {}
    unmatched_cols = []
    skipped_short = []
    skipped_nan = []

    for col in signal_cols:
        if df[col].isna().any():
            skipped_nan.append(col)
            continue

        cutoff, matched_pattern = resolve_cutoff(col, joint_group_cutoffs, default_cutoff)
        if matched_pattern is None:
            unmatched_cols.append(col)

        filtered, was_filtered = butterworth_filter(df[col].values, cutoff=cutoff)
        if not was_filtered:
            skipped_short.append(col)
            continue

        df[col] = filtered
        cutoffs_used[col] = cutoff

    if skipped_nan:
        log["warnings"].append(f"skipped_filter_cols_has_nan:{len(skipped_nan)}")
    if skipped_short:
        log["warnings"].append(f"filtering_skipped_trial_too_short:{len(skipped_short)}_cols")
    if unmatched_cols:
        log["warnings"].append(
            f"unmatched_joint_group_used_default:{len(unmatched_cols)}_cols_at_{default_cutoff}Hz"
        )

    log["filtered"] = len(cutoffs_used) > 0
    log["cutoffs_by_column"] = cutoffs_used
    log["cutoff_hz_range_used"] = (
        sorted(set(cutoffs_used.values())) if cutoffs_used else []
    )
    return df


# ---------------------------------------------------------------------------
# Combined pipeline (in-memory)
# ---------------------------------------------------------------------------
def preprocess_trial(
    filepath: str,
    joint_group_cutoffs: dict[str, float] | None = None,
    default_cutoff: float = DEFAULT_CUTOFF_HZ,
) -> tuple[pd.DataFrame, dict]:
    df, log = load_and_clean(filepath)
    df = fill_gaps(df, log)
    df = apply_filtering(df, log, joint_group_cutoffs=joint_group_cutoffs, default_cutoff=default_cutoff)
    return df, log


# ---------------------------------------------------------------------------
# Save (output folder is separate from input)
# ---------------------------------------------------------------------------
def preprocess_and_save(
    filepath: str,
    output_dir: str,
    joint_group_cutoffs: dict[str, float] | None = None,
    default_cutoff: float = DEFAULT_CUTOFF_HZ,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    df, log = preprocess_trial(filepath, joint_group_cutoffs=joint_group_cutoffs, default_cutoff=default_cutoff)

    stem = Path(filepath).stem
    out_path = os.path.join(output_dir, f"{stem}_clean.csv")
    df.to_csv(out_path, index=False)

    log["output_path"] = out_path
    log["status"] = "ok" if not any(
        "mismatch" in w or "remaining" in w for w in log["warnings"]
    ) else "warning"
    return log


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def batch_preprocess(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.csv",
    joint_group_cutoffs: dict[str, float] | None = None,
    default_cutoff: float = DEFAULT_CUTOFF_HZ,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(input_dir, "**", pattern)
    files = sorted(glob.glob(search_pattern, recursive=True))

    if not files:
        print(f"No files found matching {search_pattern}")
        return pd.DataFrame()

    logs = []
    for i, f in enumerate(files, 1):
        try:
            log = preprocess_and_save(
                f, output_dir,
                joint_group_cutoffs=joint_group_cutoffs,
                default_cutoff=default_cutoff,
            )
        except Exception as e:
            log = {"file": os.path.basename(f), "status": "error", "error": str(e), "warnings": []}
        logs.append(log)
        print(f"[{i}/{len(files)}] {log.get('file')} -> {log.get('status', 'error')}")

    # Flatten cutoffs_by_column dict into a summary-friendly string for CSV export
    for log in logs:
        if "cutoffs_by_column" in log:
            log["cutoff_hz_range_used"] = str(log.get("cutoff_hz_range_used", []))
            del log["cutoffs_by_column"]

    summary_df = pd.DataFrame(logs)
    summary_path = os.path.join(output_dir, "preprocessing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary log saved to: {summary_path}")

    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_cutoff_map(path: str) -> dict[str, float]:
    """Load a JSON file mapping regex patterns to cutoff Hz values.
    Example JSON:
    {
      "(?i)wrist|hand|finger": 10.0,
      "(?i)shoulder|elbow": 7.0,
      "(?i)trunk|spine|head": 4.0
    }
    """
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="SAMuSe MoCap preprocessing (Stages 1-3): clean, gap-fill, per-joint-group filter."
    )
    parser.add_argument("--single_file", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*.csv")
    parser.add_argument(
        "--cutoff", type=float, default=DEFAULT_CUTOFF_HZ,
        help=f"Fallback cutoff (Hz) for joints not matched by any group pattern (default: {DEFAULT_CUTOFF_HZ})"
    )
    parser.add_argument(
        "--cutoff_map", type=str, default=None,
        help="Path to JSON file with {regex_pattern: cutoff_hz} overriding built-in joint-group defaults"
    )
    parser.add_argument(
        "--use_default_groups", action="store_true", default=True,
        help="Use built-in joint-group cutoffs (wrist=10Hz, elbow=8Hz, shoulder=6Hz, trunk=4Hz, hip/leg=6Hz). "
             "Default: enabled unless --cutoff_map is provided."
    )
    args = parser.parse_args()

    if not args.single_file and not args.input_dir:
        parser.error("Provide either --single_file or --input_dir/--output_dir")
    if not args.output_dir:
        parser.error("--output_dir is required")
    if args.input_dir and os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        parser.error("--output_dir must be different from --input_dir")

    if args.cutoff_map:
        joint_group_cutoffs = load_cutoff_map(args.cutoff_map)
        print(f"Loaded custom cutoff map from {args.cutoff_map}: {joint_group_cutoffs}")
    else:
        joint_group_cutoffs = DEFAULT_JOINT_GROUP_CUTOFFS
        print(f"Using built-in joint-group cutoffs: {joint_group_cutoffs}")

    if args.single_file:
        log = preprocess_and_save(
            args.single_file, args.output_dir,
            joint_group_cutoffs=joint_group_cutoffs,
            default_cutoff=args.cutoff,
        )
        print(log)
    else:
        batch_preprocess(
            args.input_dir, args.output_dir,
            pattern=args.pattern,
            joint_group_cutoffs=joint_group_cutoffs,
            default_cutoff=args.cutoff,
        )


if __name__ == "__main__":
    main()

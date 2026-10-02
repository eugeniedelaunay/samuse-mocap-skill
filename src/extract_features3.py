"""
SAMuSe feature extraction for cleaned motion-capture trials.

Default analysis conditions: PLAY and MIME.

Extracted feature families
--------------------------
1. ROM and ROM rate for available joint-angle axes.
2. Selected duration-robust, fixed-window SPARC features:
   - Clarinet: median 2 s windowed SPARC of LWrist 3D trajectory speed.
   - Violin: median 2 s windowed SPARC of RElbowAngles_Y angular speed.
3. Stroke-segmented LDJ mean and SD:
   - Violin: RWrist trajectory speed.
   - Clarinet: RWrist and LWrist trajectory speed.
4. Selected shared PCA features imported from unified PCA output.

Deliberately excluded:
- whole-trial SPARC and whole-trial LDJ;
- SPARC mean/SD/IQR variants, extra wrist variants, and bow-direction variants;
- LDJ stroke counts;
- per-trial PCA component-count/PC1-variance features.

QC window/stroke counts are saved but are not model-ready features.

Example:
python extract_features3.py \
  --input_dir preprocessed/ \
  --output_dir features/ \
  --metadata_file metadata/participants_instrument_skill_toshare.xlsx \
  --pca_features_csv features_weiss/unified_pca_play_mime/12a_shared_pm_usage_by_block.csv \\
  features_weiss/unified_pca_play_mime/16_joint_angle_variability_family_by_block.csv
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

FRAME_RATE = 50.0
KEEP_CONDITIONS = {"play", "mime"}

WINDOW_S = 2.0
WINDOW_STEP_S = 0.1
FILTER_ORDER = 4
POSITION_CUTOFF_HZ = 10.0
SPARC_CUTOFF_HZ = 10.0
SPARC_AMP_THRESHOLD = 0.05
STROKE_PEAK_MIN_DISTANCE_S = 0.10

BASE_JOINTS = [
    "Spine1Angles", "Spine2Angles", "Spine3Angles", "Spine4Angles",
    "NeckAngles", "HeadAngles",
]
SHARED_ROM_JOINTS = BASE_JOINTS + [
    "RWristAngles", "RElbowAngles", "RShoulderAngles", "RClavicleAngles",
    "LWristAngles", "LElbowAngles", "LShoulderAngles", "LClavicleAngles",
]

INSTRUMENT_CONFIG = {
    "clarinet": {
        "rom_joints": SHARED_ROM_JOINTS,
        "selected_sparc": {"type": "wrist_position", "effector": "LWrist"},
        "ldj_effectors": ["RWrist", "LWrist"],
        "stroke": {"min_duration_s": 0.10, "min_amplitude_mm": 8.0},
    },
    "violin": {
        "rom_joints": SHARED_ROM_JOINTS,
        "selected_sparc": {"type": "bow_elbow_angle", "column": "RElbowAngles_Y (deg)"},
        "ldj_effectors": ["RWrist"],
        "stroke": {"min_duration_s": 0.15, "min_amplitude_mm": 40.0},
    },
    "default": {
        "rom_joints": SHARED_ROM_JOINTS,
        "selected_sparc": None,
        "ldj_effectors": ["RWrist"],
        "stroke": {"min_duration_s": 0.15, "min_amplitude_mm": 20.0},
    },
}

SHARED_PCA_FEATURES = [
    "PCA16_lower_body_mean_sd",
    "PCA16_trunk_head_mean_sd",
    "PCA16_upper_body_mean_sd",
    "PCA12a_PM1_rms",
    "PCA12a_PM2_rms",
    "PCA12a_PM7_rms",
]
INSTRUMENT_SPECIFIC_PCA_FEATURES = {
    "clarinet": [
        "PCA12a_PM4_rms",
    ],
    "violin": ["PCA12a_PM6_rms"],
}
PCA_MERGE_KEYS = ["participant_id", "condition", "block"]
NORMALIZE_FEATURE_PREFIXES = ("ROM_", "SPARC_", "LDJ_", "PCA_", "CRP_")


def load_instrument_map(metadata_file):
    if not metadata_file:
        return {}
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_file}")

    extension = Path(metadata_file).suffix.lower()
    if extension == ".csv":
        metadata = pd.read_csv(metadata_file)
    elif extension in {".xlsx", ".xls"}:
        metadata = pd.read_excel(metadata_file)
    else:
        raise ValueError(f"Unsupported metadata extension: {extension}")

    metadata.columns = [str(column).strip().lower() for column in metadata.columns]
    id_column = next((column for column in metadata.columns if "id" in column), None)
    instrument_column = next((column for column in metadata.columns if "instrument" in column), None)
    if id_column is None or instrument_column is None:
        raise ValueError(
            "Metadata needs participant-ID and instrument columns. "
            f"Found: {list(metadata.columns)}"
        )

    mapping = {}
    for _, row in metadata.iterrows():
        participant_id = str(row[id_column]).strip().upper()
        instrument = str(row[instrument_column]).strip().lower()
        mapping[participant_id] = instrument if instrument in INSTRUMENT_CONFIG else "default"
    return mapping


def parse_trial_metadata(filepath, instrument_map):
    stem = Path(filepath).stem.replace("_clean", "")
    match = re.match(r"(P\d+)_([a-zA-Z]+)_(\d+)", stem)
    if match is None:
        raise ValueError(f"Cannot parse P###_condition_block from: {Path(filepath).name}")
    participant_id, condition, block = match.groups()
    participant_id = participant_id.upper()
    return {
        "participant_id": participant_id,
        "condition": condition.lower(),
        "block": int(block),
        "instrument": instrument_map.get(participant_id, "default"),
        "file": Path(filepath).name,
    }


def compute_trial_duration_seconds(dataframe):
    if "time_s" in dataframe.columns:
        time = pd.to_numeric(dataframe["time_s"], errors="coerce").dropna()
        if len(time) >= 2:
            duration = float(time.max() - time.min())
            if duration > 0:
                return duration
    return float(len(dataframe) / FRAME_RATE) if len(dataframe) >= 2 else np.nan


def compute_rom(dataframe, joint_prefixes, duration):
    features = {}
    for prefix in joint_prefixes:
        for axis in ("X", "Y", "Z"):
            column = f"{prefix}_{axis} (deg)"
            if column not in dataframe.columns:
                continue
            values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
            rom_key = f"ROM_{prefix}_{axis}"
            rate_key = f"ROM_rate_{prefix}_{axis}"
            if values.empty:
                features[rom_key] = np.nan
                features[rate_key] = np.nan
                continue
            rom = float(values.max() - values.min())
            features[rom_key] = rom
            features[rate_key] = rom / duration if pd.notna(duration) and duration > 0 else np.nan
    return features


def prepare_uniform_scalar_signal(values, time_s):
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    valid = np.isfinite(values) & np.isfinite(time_s)
    values, time_s = values[valid], time_s[valid]
    if len(values) < 8:
        return None

    order = np.argsort(time_s)
    time_s, values = time_s[order], values[order]
    time_s, unique_indices = np.unique(time_s, return_index=True)
    values = values[unique_indices]
    if len(time_s) < 8:
        return None

    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    fs = 1.0 / dt
    uniform_time = np.arange(time_s[0], time_s[-1] + 0.5 * dt, dt)
    uniform_values = np.interp(uniform_time, time_s, values)
    return uniform_time, uniform_values, fs


def wrist_speed_from_position(dataframe, effector):
    position_columns = [f"{effector}Positions_{axis} (mm)" for axis in ("X", "Y", "Z")]
    if "time_s" not in dataframe.columns or not all(column in dataframe.columns for column in position_columns):
        return None

    time = pd.to_numeric(dataframe["time_s"], errors="coerce").to_numpy(dtype=float)
    positions = dataframe[position_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(positions).all(axis=1)
    time, positions = time[valid], positions[valid]
    if len(time) < 8:
        return None

    order = np.argsort(time)
    time, positions = time[order], positions[order]
    time, unique_indices = np.unique(time, return_index=True)
    positions = positions[unique_indices]
    if len(time) < 8:
        return None

    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    fs = 1.0 / dt
    uniform_time = np.arange(time[0], time[-1] + 0.5 * dt, dt)
    uniform_positions = np.column_stack([
        np.interp(uniform_time, time, positions[:, axis]) for axis in range(3)
    ])
    velocity = np.gradient(uniform_positions, uniform_time, axis=0)
    try:
        sos = butter(FILTER_ORDER, POSITION_CUTOFF_HZ, btype="lowpass", fs=fs, output="sos")
        filtered_velocity = sosfiltfilt(sos, velocity, axis=0)
    except ValueError:
        return None
    return np.linalg.norm(filtered_velocity, axis=1), fs


def angular_speed_from_angle(dataframe, angle_column):
    if "time_s" not in dataframe.columns or angle_column not in dataframe.columns:
        return None
    prepared = prepare_uniform_scalar_signal(
        pd.to_numeric(dataframe[angle_column], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(dataframe["time_s"], errors="coerce").to_numpy(dtype=float),
    )
    if prepared is None:
        return None
    uniform_time, angle, fs = prepared
    angular_velocity = np.gradient(angle, uniform_time)
    try:
        sos = butter(FILTER_ORDER, POSITION_CUTOFF_HZ, btype="lowpass", fs=fs, output="sos")
        filtered_velocity = sosfiltfilt(sos, angular_velocity)
    except ValueError:
        return None
    return np.abs(filtered_velocity), fs


def sparc(speed, fs):
    speed = np.asarray(speed, dtype=float)
    if len(speed) < 4 or not np.isfinite(speed).all() or np.max(np.abs(speed)) <= 0:
        return np.nan

    nfft = int(2 ** np.ceil(np.log2(len(speed))) + 4)
    frequency = np.arange(nfft) * fs / nfft
    spectrum = np.abs(np.fft.fft(speed, nfft))
    if spectrum.max() <= 0:
        return np.nan
    spectrum = spectrum / spectrum.max()

    mask = frequency <= SPARC_CUTOFF_HZ
    frequency, spectrum = frequency[mask], spectrum[mask]
    usable = np.where(spectrum >= SPARC_AMP_THRESHOLD)[0]
    if len(usable) < 2:
        return np.nan

    frequency = frequency[usable[0]:usable[-1] + 1]
    spectrum = spectrum[usable[0]:usable[-1] + 1]
    frequency_range = frequency[-1] - frequency[0]
    if len(frequency) < 2 or frequency_range <= 0:
        return np.nan

    return float(-np.sum(np.sqrt(
        (np.diff(frequency) / frequency_range) ** 2 + np.diff(spectrum) ** 2
    )))


def fixed_window_sparc_median(speed, fs):
    window_n = int(round(WINDOW_S * fs))
    step_n = max(1, int(round(WINDOW_STEP_S * fs)))
    if len(speed) < window_n:
        return np.nan, 0

    values = []
    for start in range(0, len(speed) - window_n + 1, step_n):
        value = sparc(speed[start:start + window_n], fs)
        if np.isfinite(value):
            values.append(value)
    return (float(np.median(values)), len(values)) if values else (np.nan, 0)


def compute_selected_windowed_sparc(dataframe, instrument):
    config = INSTRUMENT_CONFIG.get(instrument, INSTRUMENT_CONFIG["default"])
    selected = config["selected_sparc"]
    if selected is None:
        return {}

    if selected["type"] == "wrist_position":
        result = wrist_speed_from_position(dataframe, selected["effector"])
        if result is None:
            return {"SPARC_window_median_LWrist": np.nan, "QC_SPARC_window_n_LWrist": 0}
        speed, fs = result
        median, n_windows = fixed_window_sparc_median(speed, fs)
        return {"SPARC_window_median_LWrist": median, "QC_SPARC_window_n_LWrist": n_windows}

    if selected["type"] == "bow_elbow_angle":
        result = angular_speed_from_angle(dataframe, selected["column"])
        if result is None:
            return {
                "SPARC_window_bow_elbow_all_median": np.nan,
                "QC_SPARC_window_bow_elbow_all_n": 0,
            }
        speed, fs = result
        median, n_windows = fixed_window_sparc_median(speed, fs)
        return {
            "SPARC_window_bow_elbow_all_median": median,
            "QC_SPARC_window_bow_elbow_all_n": n_windows,
        }

    return {}


def segment_strokes(speed, fs, instrument):
    params = INSTRUMENT_CONFIG.get(instrument, INSTRUMENT_CONFIG["default"])["stroke"]
    if speed is None or len(speed) < int(params["min_duration_s"] * fs) + 2:
        return []

    minima, _ = find_peaks(
        -speed,
        distance=max(1, int(STROKE_PEAK_MIN_DISTANCE_S * fs)),
    )
    boundaries = sorted(set([0, *minima.tolist(), len(speed) - 1]))
    strokes = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        duration = (end - start) / fs
        path_length = float(np.sum(speed[start:end]) / fs)
        if duration >= params["min_duration_s"] and path_length >= params["min_amplitude_mm"]:
            strokes.append((start, end))
    return strokes


def ldj(speed, fs):
    """Log dimensionless jerk calculated only on individual stroke windows."""
    if speed is None or len(speed) < 4:
        return np.nan
    speed = np.asarray(speed, dtype=float)
    if not np.isfinite(speed).all():
        return np.nan

    peak_speed = np.max(np.abs(speed))
    if peak_speed <= 0:
        return np.nan
    jerk = np.diff(speed, n=2) * fs ** 2
    if len(jerk) < 1:
        return np.nan
    jerk_integral = np.sum(jerk ** 2) / fs
    dimensionless_jerk = (len(speed) / fs) ** 3 * jerk_integral / peak_speed ** 2
    if dimensionless_jerk <= 0 or not np.isfinite(dimensionless_jerk):
        return np.nan
    return float(-np.log(dimensionless_jerk))


def compute_stroke_ldj_features(dataframe, instrument):
    config = INSTRUMENT_CONFIG.get(instrument, INSTRUMENT_CONFIG["default"])
    results = {}
    for effector in config["ldj_effectors"]:
        mean_key = f"LDJ_stroke_mean_{effector}"
        sd_key = f"LDJ_stroke_sd_{effector}"
        qc_key = f"QC_LDJ_stroke_n_{effector}"
        speed_result = wrist_speed_from_position(dataframe, effector)
        if speed_result is None:
            results[mean_key] = np.nan
            results[sd_key] = np.nan
            results[qc_key] = 0
            continue

        speed, fs = speed_result
        values = []
        for start, end in segment_strokes(speed, fs, instrument):
            value = ldj(speed[start:end], fs)
            if np.isfinite(value):
                values.append(value)

        results[mean_key] = float(np.mean(values)) if values else np.nan
        results[sd_key] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        results[qc_key] = len(values)
    return results


def extract_trial_features(filepath, instrument_map):
    dataframe = pd.read_csv(filepath)
    metadata = parse_trial_metadata(filepath, instrument_map)
    config = INSTRUMENT_CONFIG.get(metadata["instrument"], INSTRUMENT_CONFIG["default"])
    duration = compute_trial_duration_seconds(dataframe)
    return {
        **metadata,
        "n_rows": len(dataframe),
        "trial_duration_seconds": duration,
        **compute_rom(dataframe, config["rom_joints"], duration),
        **compute_selected_windowed_sparc(dataframe, metadata["instrument"]),
        **compute_stroke_ldj_features(dataframe, metadata["instrument"]),
    }


def normalize_features_by_instrument(dataframe):
    if "instrument" not in dataframe.columns:
        return dataframe
    dataframe = dataframe.copy()
    feature_columns = [
        column for column in dataframe.columns
        if column.startswith(NORMALIZE_FEATURE_PREFIXES)
        and not column.startswith("QC_")
        and not column.endswith("_z")
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    def zscore(group):
        values = group.to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        if len(valid) < 2:
            return pd.Series(np.nan, index=group.index)
        standard_deviation = np.std(valid)
        if standard_deviation == 0 or not np.isfinite(standard_deviation):
            return pd.Series(np.nan, index=group.index)
        return (group - np.mean(valid)) / standard_deviation

    for column in feature_columns:
        dataframe[f"{column}_z"] = dataframe.groupby("instrument")[column].transform(zscore)
    return dataframe


def standardize_merge_keys(dataframe, keys, source_name):
    dataframe = dataframe.copy()
    missing = [key for key in keys if key not in dataframe.columns]
    if missing:
        raise ValueError(f"{source_name} is missing merge key(s): {missing}")

    dataframe["participant_id"] = dataframe["participant_id"].astype(str).str.strip().str.upper()
    dataframe["condition"] = dataframe["condition"].astype(str).str.strip().str.lower()
    if "block" in keys:
        dataframe["block"] = pd.to_numeric(dataframe["block"], errors="coerce").astype("Int64")
    if dataframe[keys].isna().any().any():
        examples = dataframe.loc[dataframe[keys].isna().any(axis=1), keys].head(10)
        raise ValueError(f"{source_name} has invalid merge key values:\n{examples.to_string(index=False)}")
    return dataframe


def merge_selected_pca_features(features, pca_features_csv, output_dir, include_instrument_specific=False):
    """Merge block-level PCA features on participant, condition and block.

    Every PCA file must contain a block column. A participant/condition-level
    merge would give identical values to all blocks of a participant, so
    the feature could identify the participant under trial-wise folds.
    """
    if not pca_features_csv:
        return features
    paths = [pca_features_csv] if isinstance(pca_features_csv, str) else list(pca_features_csv)

    selected = list(SHARED_PCA_FEATURES)
    if include_instrument_specific:
        selected.extend(
            feature for feature_list in INSTRUMENT_SPECIFIC_PCA_FEATURES.values()
            for feature in feature_list
        )

    merged = standardize_merge_keys(features, PCA_MERGE_KEYS, "Extracted features")
    overlap = [feature for feature in selected if feature in merged.columns]
    if overlap:
        raise ValueError(f"PCA feature columns already exist in the extracted table: {overlap}")

    found = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PCA feature file does not exist: {path}")
        right = pd.read_csv(path)
        if "block" not in right.columns:
            raise ValueError(
                f"{path} has no 'block' column. Merging on participant/condition only would "
                "assign identical PCA values to all blocks of a participant (participant-identity leak)."
            )
        right = standardize_merge_keys(right, PCA_MERGE_KEYS, os.path.basename(path))
        columns = [feature for feature in selected if feature in right.columns and feature not in found]
        if not columns:
            print(f"PCA file {os.path.basename(path)}: no requested features; skipped.")
            continue
        if right.duplicated(PCA_MERGE_KEYS).any():
            duplicates = right.loc[right.duplicated(PCA_MERGE_KEYS, keep=False), PCA_MERGE_KEYS].head(10)
            raise ValueError(
                f"{path} has duplicate merge keys; merge is ambiguous:\n{duplicates.to_string(index=False)}"
            )
        merged = merged.merge(
            right[PCA_MERGE_KEYS + columns],
            on=PCA_MERGE_KEYS,
            how="left",
            validate="many_to_one",
            indicator="_pca_merge",
        )
        matched = int((merged["_pca_merge"] == "both").sum())
        print(f"PCA merge ({os.path.basename(path)}): matched {matched}/{len(merged)} rows; added {len(columns)} feature(s).")
        unmatched = merged.loc[merged["_pca_merge"] != "both", PCA_MERGE_KEYS]
        if not unmatched.empty:
            out = os.path.join(output_dir, f"unmatched_pca_rows_{Path(path).stem}.csv")
            unmatched.to_csv(out, index=False)
            print(f"Warning: saved {len(unmatched)} unmatched PCA rows to: {out}")
        merged = merged.drop(columns="_pca_merge")
        found.extend(columns)

    missing = [feature for feature in selected if feature not in found]
    if missing:
        raise ValueError(f"Requested PCA features not found in any PCA file: {missing}")

    if include_instrument_specific:
        instrument = merged["instrument"].astype(str).str.strip().str.lower()
        for name, instrument_features in INSTRUMENT_SPECIFIC_PCA_FEATURES.items():
            for feature in instrument_features:
                merged.loc[instrument != name, feature] = np.nan

    return merged


def filter_files_by_condition(files, instrument_map, keep_conditions):
    kept = []
    skipped = 0
    for filepath in files:
        try:
            metadata = parse_trial_metadata(filepath, instrument_map)
        except ValueError as error:
            print(f"Skipping unparseable file: {error}")
            continue
        if keep_conditions is None or metadata["condition"] in keep_conditions:
            kept.append(filepath)
        else:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} file(s) outside requested conditions.")
    return kept


def batch_extract_features(
    input_dir,
    output_dir,
    instrument_map,
    pattern,
    keep_conditions,
    pca_features_csv,
    include_instrument_specific_pca,
    normalize,
):
    os.makedirs(output_dir, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
    files = filter_files_by_condition(all_files, instrument_map, keep_conditions)
    if not files:
        raise FileNotFoundError(f"No usable files found under {input_dir} matching {pattern}")

    rows = []
    for index, filepath in enumerate(files, start=1):
        try:
            row = extract_trial_features(filepath, instrument_map)
            row["status"] = "ok"
        except Exception as error:
            row = {"file": Path(filepath).name, "status": "error", "error": str(error)}
        rows.append(row)
        print(f"[{index}/{len(files)}] {Path(filepath).name}: {row['status']}")

    features = pd.DataFrame(rows)
    successful = features.loc[features["status"].eq("ok")].copy()
    failures = features.loc[~features["status"].eq("ok")].copy()

    successful = merge_selected_pca_features(
        successful,
        pca_features_csv,
        output_dir,
        include_instrument_specific_pca,
    )
    if normalize:
        successful = normalize_features_by_instrument(successful)

    features = pd.concat([successful, failures], ignore_index=True, sort=False)
    output_path = os.path.join(output_dir, "trial_features.csv")
    features.to_csv(output_path, index=False)
    print(f"Saved trial-level features: {output_path}")
    return features


def aggregate_cv(trial_features_path, output_dir):
    dataframe = pd.read_csv(trial_features_path)
    if "status" in dataframe.columns:
        dataframe = dataframe[dataframe["status"] == "ok"].copy()

    metrics = [
        column for column in dataframe.columns
        if column.startswith(("ROM_", "SPARC_", "LDJ_", "PCA_", "CRP_"))
        and not column.startswith("QC_")
    ]
    rows = []
    for (participant_id, condition), group in dataframe.groupby(["participant_id", "condition"]):
        row = {"participant_id": participant_id, "condition": condition, "n_trials": len(group)}
        if "instrument" in group.columns:
            row["instrument"] = group["instrument"].iloc[0]
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"CV_{metric}"] = (
                float(np.std(values) / np.mean(values))
                if len(values) >= 2 and np.mean(values) != 0
                else np.nan
            )
        rows.append(row)

    output = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, "cv_features.csv")
    output.to_csv(output_path, index=False)
    print(f"Saved CV features: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="SAMuSe feature extraction with selected windowed SPARC, stroke LDJ, and shared PCA features."
    )
    parser.add_argument("--input_dir", default=None, help="Directory containing cleaned trial CSV files")
    parser.add_argument("--single_file", default=None, help="One cleaned trial CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--metadata_file", default=None, help="Participant metadata CSV/XLSX")
    parser.add_argument("--pattern", default="*_clean.csv", help="Recursive batch file pattern")
    parser.add_argument("--conditions", default="play,mime", help="Comma-separated conditions; use all for no filter")
    parser.add_argument("--pca_features_csv", nargs="+", default=None, help="One or more block-level PCA feature CSVs (must contain participant_id, condition, block)")
    parser.add_argument(
        "--include_instrument_specific_pca",
        action="store_true",
        help="Also merge clarinet-only PCA11b/PM4 and violin-only PM6 candidates.",
    )
    parser.add_argument("--no_normalize", action="store_true", help="Do not add within-instrument *_z columns")
    parser.add_argument("--aggregate_cv", action="store_true", help="Aggregate existing trial_features.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.aggregate_cv:
        if not args.input_dir:
            parser.error("--aggregate_cv requires --input_dir containing trial_features.csv")
        aggregate_cv(os.path.join(args.input_dir, "trial_features.csv"), args.output_dir)
        return

    instrument_map = load_instrument_map(args.metadata_file) if args.metadata_file else {}
    if not instrument_map:
        print("Warning: no instrument metadata loaded; default configuration will be used.")

    if args.single_file:
        row = extract_trial_features(args.single_file, instrument_map)
        row["status"] = "ok"
        result = pd.DataFrame([row])
        result = merge_selected_pca_features(
            result,
            args.pca_features_csv,
            args.output_dir,
            args.include_instrument_specific_pca,
        )
        if not args.no_normalize:
            result = normalize_features_by_instrument(result)
        output_path = os.path.join(args.output_dir, "single_trial_features.csv")
        result.to_csv(output_path, index=False)
        print(result.to_string(index=False))
        return

    if not args.input_dir:
        parser.error("Provide --input_dir for batch extraction or --single_file for one file")
    conditions = None if args.conditions.strip().lower() == "all" else {
        value.strip().lower() for value in args.conditions.split(",") if value.strip()
    }
    batch_extract_features(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        instrument_map=instrument_map,
        pattern=args.pattern,
        keep_conditions=conditions,
        pca_features_csv=args.pca_features_csv,
        include_instrument_specific_pca=args.include_instrument_specific_pca,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()

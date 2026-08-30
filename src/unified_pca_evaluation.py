"""
unified_pca_evaluation.py -- SAMuSe unified evaluation of four movement
feature implementations.

Implements and evaluates, separately rather than mixing their meanings:

A. Existing PCA baseline from extract_features3.py
   - PCA_n_components_90
   - PCA_total_variance_explained_by_pc1

B. 11b-style coordination consistency
   - fit a separate standardized PCA to each block
   - retain PC1 loading vector
   - sign-align PC1 loadings within participant x condition
   - calculate mean pairwise cosine similarity across blocks

C. 12a-style shared Principal Movements (PMs)
   - one shared PCA fitted on every selected block stacked together
   - Bigand-style normalization per block: de-mean each signal by its own
     block posture, then divide the full block by ONE global scalar SD
   - PMs are fixed across all selected participants/blocks
   - save all loading vectors, cumulative variance, and per-block PM RMS
   - follow 12a defaults: 95% variance target, 25 PMs saved/analysed,
     and first 7 PM RMS features shown/evaluated in summary output

D. 16-style joint-angle movement variability
   - temporal SD per block x curated angle signal
   - participant-level mean SD per signal and grouped family summaries

The script evaluates candidate participant-level features on selected
condition(s), separately for violin and clarinet:
- participant-level aggregation across blocks
- skill-group ANOVA and Kruskal-Wallis test
- correlation with mean trial duration
- skill test after duration residualization
- redundancy correlations with baseline PCA / ROM / SPARC / LDJ features

The default is condition=play. This is recommended for the first analysis:
PCA/PM/variability measures have their clearest movement interpretation
when people physically perform.

Shared full-body set: 50 validated angle channels, common to violin and
clarinet. Locked/near-zero / end-joint / duplicate channels are excluded.

Usage:
python src/unified_pca_evaluation.py \
  --input_dir preprocessed \
  --trial_features features_weiss/trial_features.csv \
  --metadata_file metadata/participants_metadata_clean.csv \
  --output_dir features_weiss/unified_pca_play \
  --condition play
"""

import argparse
import glob
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Frozen, shared full-body set: 50 validated joint-angle signals.
# ---------------------------------------------------------------------------
CURATED_FULL_BODY_ANGLE_COLUMNS = [
    # Lower body
    "LHipAngles_X (deg)", "LHipAngles_Y (deg)", "LHipAngles_Z (deg)",
    "RHipAngles_X (deg)", "RHipAngles_Y (deg)", "RHipAngles_Z (deg)",
    "LKneeAngles_X (deg)", "LKneeAngles_Y (deg)", "LKneeAngles_Z (deg)",
    "RKneeAngles_X (deg)", "RKneeAngles_Y (deg)", "RKneeAngles_Z (deg)",
    "LAnkleAngles_X (deg)", "LAnkleAngles_Y (deg)", "LAnkleAngles_Z (deg)",
    "RAnkleAngles_X (deg)", "RAnkleAngles_Y (deg)", "RAnkleAngles_Z (deg)",
    # Trunk, neck, head
    "Spine1Angles_X (deg)", "Spine1Angles_Y (deg)", "Spine1Angles_Z (deg)",
    "Spine2Angles_X (deg)", "Spine2Angles_Y (deg)", "Spine2Angles_Z (deg)",
    "Spine3Angles_X (deg)", "Spine3Angles_Y (deg)", "Spine3Angles_Z (deg)",
    "NeckAngles_X (deg)", "NeckAngles_Y (deg)", "NeckAngles_Z (deg)",
    "HeadAngles_X (deg)", "HeadAngles_Y (deg)", "HeadAngles_Z (deg)",
    # Upper body
    "LShoulderAngles_X (deg)", "LShoulderAngles_Y (deg)", "LShoulderAngles_Z (deg)",
    "RShoulderAngles_X (deg)", "RShoulderAngles_Y (deg)", "RShoulderAngles_Z (deg)",
    "LClavicleAngles_X (deg)", "LClavicleAngles_Y (deg)", "LClavicleAngles_Z (deg)",
    "RClavicleAngles_X (deg)", "RClavicleAngles_Y (deg)", "RClavicleAngles_Z (deg)",
    "LElbowAngles_Y (deg)", "RElbowAngles_Y (deg)",
    "LWristAngles_X (deg)", "LWristAngles_Y (deg)", "LWristAngles_Z (deg)",
    "RWristAngles_X (deg)", "RWristAngles_Y (deg)", "RWristAngles_Z (deg)",
]

FAMILY_PREFIXES = {
    "lower_body": ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"),
    "trunk_head": ("Spine1", "Spine2", "Spine3", "Neck", "Head"),
    "upper_body": ("LShoulder", "RShoulder", "LClavicle", "RClavicle", "LElbow", "RElbow", "LWrist", "RWrist"),
}

BASELINE_FEATURES = [
    "PCA_n_components_90",
    "PCA_total_variance_explained_by_pc1",
]
RELATED_FEATURES = [
    "SPARC", "SPARC_RWrist", "SPARC_LWrist",
    "SPARC_stroke_mean_RWrist", "SPARC_stroke_sd_RWrist",
    "SPARC_stroke_mean_LWrist", "SPARC_stroke_sd_LWrist",
    "LDJ_stroke_mean_RWrist", "LDJ_stroke_sd_RWrist",
    "LDJ_stroke_mean_LWrist", "LDJ_stroke_sd_LWrist",
]

VARIANCE_TARGETS = (0.90, 0.95)
NB_PM_DETAIL = 25       # exact 12a default
N_PM_TO_SHOW = 7        # exact 12a default plot/evaluation focus
HIGH_REDUNDANCY = 0.90
FRAME_RATE = 50.0


def parse_trial_metadata(filepath):
    stem = Path(filepath).stem.replace("_clean", "")
    match = re.match(r"(P\d+)_([a-zA-Z]+)_(\d+)", stem)
    if not match:
        return None
    participant, condition, block = match.groups()
    return {"participant_id": participant.upper(), "condition": condition.lower(), "block": int(block), "file": os.path.basename(filepath)}


def load_metadata(metadata_file):
    if not metadata_file or not os.path.exists(metadata_file):
        return pd.DataFrame(columns=["participant_id"])
    meta = pd.read_csv(metadata_file)
    meta.columns = [str(c).strip() for c in meta.columns]
    id_col = next((c for c in meta.columns if c.lower() == "id"), meta.columns[0])
    meta = meta.rename(columns={id_col: "participant_id"})
    meta["participant_id"] = meta["participant_id"].astype(str).str.strip().str.upper()
    return meta


def load_trial_features(path, metadata):
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    df["participant_id"] = df["participant_id"].astype(str).str.strip().str.upper()
    return df.merge(metadata, on="participant_id", how="left", suffixes=("", "_meta"))


def load_block_matrix(filepath):
    df = pd.read_csv(filepath)
    missing = [c for c in CURATED_FULL_BODY_ANGLE_COLUMNS if c not in df.columns]
    if missing:
        return None, f"missing {len(missing)} curated columns"
    matrix = df[CURATED_FULL_BODY_ANGLE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if len(matrix) < 10:
        return None, "fewer than 10 valid frames"
    return matrix, None


def normalize_shared_pm_block(matrix):
    """12a/Bigand-style: remove own mean posture then divide by one global SD."""
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    global_sd = float(np.std(centered))
    if not np.isfinite(global_sd) or global_sd <= 0:
        return None
    return centered / global_sd


def pc1_loading_per_block(matrix):
    """11b-style: standardize each channel within block, fit PCA PC1."""
    channel_sd = np.std(matrix, axis=0)
    variable = channel_sd > 1e-9
    if variable.sum() < 2:
        return None
    standardized = StandardScaler().fit_transform(matrix[:, variable])
    try:
        pca = PCA(n_components=1).fit(standardized)
    except ValueError:
        return None
    full = np.zeros(matrix.shape[1])
    full[variable] = pca.components_[0]
    return full


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else np.nan


def sign_align(vectors):
    if not vectors:
        return []
    reference = vectors[0]
    return [v if np.dot(v, reference) >= 0 else -v for v in vectors]


def trial_duration(df):
    if "time_s" in df.columns:
        values = pd.to_numeric(df["time_s"], errors="coerce").dropna()
        if len(values) >= 2:
            duration = float(values.max() - values.min())
            if duration > 0:
                return duration
    return len(df) / FRAME_RATE


def collect_blocks(input_dir, conditions, metadata):
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*_clean.csv"), recursive=True))
    metadata_map = metadata.set_index("participant_id").to_dict(orient="index") if not metadata.empty else {}
    records = []
    for index, filepath in enumerate(files, 1):
        trial = parse_trial_metadata(filepath)
        if trial is None or trial["condition"] not in conditions:
            continue
        participant_info = metadata_map.get(trial["participant_id"], {})
        instrument = str(participant_info.get("Instrument", participant_info.get("instrument", ""))).strip().lower()
        if instrument not in {"violin", "clarinet"}:
            continue
        raw = pd.read_csv(filepath)
        matrix, error = load_block_matrix(filepath)
        if matrix is None:
            print(f"Skipping {trial['file']}: {error}")
            continue
        record = {
            **trial,
            "instrument": instrument,
            "skill_level": participant_info.get("Skill Level", participant_info.get("skill_level", np.nan)),
            "duration_s": trial_duration(raw),
            "matrix": matrix,
            "n_frames": len(matrix),
        }
        records.append(record)
        if index % 50 == 0:
            print(f"Loaded {len(records)} usable selected blocks...")
    return records


def compute_11b_consistency(records):
    rows = []
    grouped = {}
    for record in records:
        key = (record["participant_id"], record["instrument"], record["condition"], record["skill_level"])
        loading = pc1_loading_per_block(record["matrix"])
        if loading is not None:
            grouped.setdefault(key, []).append((record, loading))
    for key, items in grouped.items():
        if len(items) < 2:
            continue
        aligned = sign_align([loading for _, loading in items])
        similarities = [cosine_similarity(a, b) for a, b in combinations(aligned, 2)]
        participant, instrument, condition, skill_level = key
        rows.append({
            "participant_id": participant,
            "instrument": instrument,
            "condition": condition,
            "Skill Level": skill_level,
            "n_blocks": len(items),
            "mean_duration_s": float(np.mean([record["duration_s"] for record, _ in items])),
            "PCA11b_within_participant_consistency": float(np.nanmean(similarities)),
            "PCA11b_consistency_sd_pairs": float(np.nanstd(similarities, ddof=1)) if len(similarities) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def compute_12a_shared_pms(records):
    normalized_blocks = []
    usable_records = []
    for record in records:
        normalized = normalize_shared_pm_block(record["matrix"])
        if normalized is not None:
            normalized_blocks.append(normalized)
            usable_records.append(record)
    if not normalized_blocks:
        raise ValueError("No blocks available for shared PM PCA.")

    stacked = np.vstack(normalized_blocks)
    pca = PCA().fit(stacked)
    scores = pca.transform(stacked)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    counts = {target: int(np.argmax(cumulative >= target) + 1) for target in VARIANCE_TARGETS}

    loading_df = pd.DataFrame(pca.components_, columns=CURATED_FULL_BODY_ANGLE_COLUMNS)
    loading_df.insert(0, "PM", [f"PM{i + 1}" for i in range(len(loading_df))])
    loading_df.insert(1, "variance_explained", pca.explained_variance_ratio_)
    loading_df.insert(2, "cumulative_variance_explained", cumulative)

    block_rows = []
    cursor = 0
    n_pm = min(NB_PM_DETAIL, scores.shape[1])
    for record, normalized in zip(usable_records, normalized_blocks):
        n_frames = len(normalized)
        block_scores = scores[cursor:cursor + n_frames, :n_pm]
        row = {key: record[key] for key in ["participant_id", "instrument", "condition", "block", "skill_level", "duration_s", "n_frames"]}
        for pm_index in range(n_pm):
            row[f"PCA12a_PM{pm_index + 1}_rms"] = float(np.sqrt(np.mean(block_scores[:, pm_index] ** 2)))
        block_rows.append(row)
        cursor += n_frames
    return loading_df, pd.DataFrame(block_rows), counts


def compute_16_variability(records):
    rows = []
    for record in records:
        sd_values = np.std(record["matrix"], axis=0, ddof=1)
        for signal, value in zip(CURATED_FULL_BODY_ANGLE_COLUMNS, sd_values):
            family = next((name for name, prefixes in FAMILY_PREFIXES.items() if signal.startswith(prefixes)), "other")
            rows.append({
                "participant_id": record["participant_id"], "instrument": record["instrument"],
                "condition": record["condition"], "block": record["block"],
                "skill_level": record["skill_level"], "duration_s": record["duration_s"],
                "signal": signal, "family": family, "PCA16_temporal_sd_deg": float(value),
            })
    signal_df = pd.DataFrame(rows)
    family_df = signal_df.groupby(
        ["participant_id", "instrument", "condition", "block", "skill_level", "duration_s", "family"],
        as_index=False,
    )["PCA16_temporal_sd_deg"].mean()
    family_df = family_df.pivot_table(
        index=["participant_id", "instrument", "condition", "block", "skill_level", "duration_s"],
        columns="family", values="PCA16_temporal_sd_deg",
    ).reset_index()
    family_df = family_df.rename(columns={family: f"PCA16_{family}_mean_sd" for family in FAMILY_PREFIXES})
    return signal_df, family_df


def aggregate_block_features(block_df, skill_col="skill_level"):
    fixed = ["participant_id", "instrument", "condition"]
    if skill_col in block_df.columns:
        fixed.append(skill_col)
    numeric = [
        c for c in block_df.columns
        if c not in fixed + ["block", "n_frames"] and pd.api.types.is_numeric_dtype(block_df[c])
    ]
    aggregation = {"block": "nunique"}
    for column in numeric:
        aggregation[column] = "mean"
    participant = block_df.groupby(fixed, as_index=False).agg(aggregation).rename(columns={"block": "n_blocks"})
    return participant


def extract_baseline_participant_features(trial_features, conditions):
    required = ["participant_id", "instrument", "condition", "block", "trial_duration_seconds"]
    if not all(c in trial_features.columns for c in required):
        return pd.DataFrame()
    sub = trial_features[trial_features["condition"].astype(str).str.lower().isin(conditions)].copy()
    if "Skill Level" not in sub.columns:
        sub["Skill Level"] = np.nan
    available = [c for c in BASELINE_FEATURES + RELATED_FEATURES if c in sub.columns]
    keep = ["participant_id", "instrument", "condition", "block", "Skill Level", "trial_duration_seconds"] + available
    sub = sub[keep].copy()
    sub = sub.rename(columns={"trial_duration_seconds": "mean_duration_s"})
    return aggregate_block_features(sub.rename(columns={"Skill Level": "skill_level"}), skill_col="skill_level")


def eta_squared(groups):
    all_values = np.concatenate(groups)
    total = np.sum((all_values - np.mean(all_values)) ** 2)
    between = sum(len(group) * (np.mean(group) - np.mean(all_values)) ** 2 for group in groups)
    return float(between / total) if total > 0 else np.nan


def skill_duration_evaluation(participant_df, feature_cols):
    rows = []
    for instrument, subset_inst in participant_df.groupby("instrument"):
        for feature in feature_cols:
            if feature not in subset_inst.columns:
                continue
            subset = subset_inst.dropna(subset=[feature, "mean_duration_s", "skill_level"])
            if len(subset) < 10 or subset["skill_level"].nunique() < 2:
                continue
            x = subset[feature].to_numpy(dtype=float)
            duration = subset["mean_duration_s"].to_numpy(dtype=float)
            try:
                r, p_duration = stats.pearsonr(x, duration)
            except Exception:
                r, p_duration = np.nan, np.nan
            groups = [group[feature].to_numpy(dtype=float) for _, group in subset.groupby("skill_level")]
            groups = [group for group in groups if len(group) >= 2]
            try:
                f_raw, p_raw = stats.f_oneway(*groups) if len(groups) >= 2 else (np.nan, np.nan)
            except Exception:
                f_raw, p_raw = np.nan, np.nan
            # duration residualization: least-squares feature ~ duration
            design = np.column_stack([np.ones(len(duration)), duration])
            coefficients, _, _, _ = np.linalg.lstsq(design, x, rcond=None)
            residual = x - design @ coefficients
            residual_groups = [residual[subset["skill_level"].to_numpy() == level] for level in subset["skill_level"].unique()]
            residual_groups = [group for group in residual_groups if len(group) >= 2]
            try:
                f_res, p_res = stats.f_oneway(*residual_groups) if len(residual_groups) >= 2 else (np.nan, np.nan)
            except Exception:
                f_res, p_res = np.nan, np.nan
            rows.append({
                "instrument": instrument, "feature": feature, "n_participants": len(subset),
                "duration_pearson_r": r, "duration_pearson_p": p_duration,
                "raw_skill_anova_F": f_raw, "raw_skill_anova_p": p_raw,
                "raw_skill_eta_squared": eta_squared(groups) if len(groups) >= 2 else np.nan,
                "duration_residual_skill_anova_F": f_res,
                "duration_residual_skill_anova_p": p_res,
            })
    return pd.DataFrame(rows)


def redundancy_evaluation(participant_df, candidate_features):
    rows = []
    available = [c for c in candidate_features if c in participant_df.columns]
    for instrument, subset_inst in participant_df.groupby("instrument"):
        for first, second in combinations(available, 2):
            sub = subset_inst[[first, second]].dropna()
            if len(sub) < 5 or sub[first].std() == 0 or sub[second].std() == 0:
                continue
            r, p = stats.pearsonr(sub[first], sub[second])
            rows.append({
                "instrument": instrument, "feature_1": first, "feature_2": second,
                "n_participants": len(sub), "pearson_r": r, "pearson_p": p,
                "flag_high_redundancy_abs_r_ge_0_90": abs(r) >= HIGH_REDUNDANCY,
            })
    result = pd.DataFrame(rows)
    return result.sort_values("pearson_r", key=lambda values: values.abs(), ascending=False) if not result.empty else result


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation of baseline PCA, 11b, 12a, and 16 movement analyses.")
    parser.add_argument("--input_dir", required=True, help="Directory containing *_clean.csv files")
    parser.add_argument("--trial_features", required=True, help="features_weiss/trial_features.csv")
    parser.add_argument("--metadata_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition", nargs="+", default=["play"], help="Conditions to analyse; default play")
    args = parser.parse_args()

    conditions = [c.lower() for c in args.condition]
    os.makedirs(args.output_dir, exist_ok=True)
    metadata = load_metadata(args.metadata_file)
    trial_features = load_trial_features(args.trial_features, metadata)

    print(f"Using {len(CURATED_FULL_BODY_ANGLE_COLUMNS)} shared full-body angle signals.")
    print(f"Conditions: {conditions}")
    print("\nLoading clean motion-capture blocks...")
    records = collect_blocks(args.input_dir, conditions, metadata)
    if not records:
        raise ValueError("No valid blocks loaded. Check paths, metadata and condition names.")
    print(f"Loaded {len(records)} blocks from {len(set(r['participant_id'] for r in records))} participants.")

    # 11b
    print("\n[1/4] 11b-style coordination consistency...")
    consistency = compute_11b_consistency(records)
    consistency.to_csv(os.path.join(args.output_dir, "11b_coordination_consistency_participant.csv"), index=False)

    # 12a
    print("[2/4] 12a-style shared Principal Movements...")
    pm_loadings, pm_block_usage, pm_counts = compute_12a_shared_pms(records)
    pm_loadings.to_csv(os.path.join(args.output_dir, "12a_shared_pm_loadings_all.csv"), index=False)
    pm_block_usage.to_csv(os.path.join(args.output_dir, "12a_shared_pm_usage_by_block.csv"), index=False)
    pd.DataFrame([{
        "n_curated_signals": len(CURATED_FULL_BODY_ANGLE_COLUMNS),
        "n_blocks": len(pm_block_usage),
        "n_pm_90": pm_counts[0.90],
        "n_pm_95": pm_counts[0.95],
        "n_pm_detail_saved": min(NB_PM_DETAIL, len(pm_loadings)),
        "n_pm_primary_evaluation": min(N_PM_TO_SHOW, len(pm_loadings)),
    }]).to_csv(os.path.join(args.output_dir, "12a_shared_pm_summary.csv"), index=False)
    pm_participant = aggregate_block_features(pm_block_usage, skill_col="skill_level")
    pm_participant.to_csv(os.path.join(args.output_dir, "12a_shared_pm_usage_participant.csv"), index=False)

    # 16
    print("[3/4] 16-style joint-angle movement variability...")
    variability_signal, variability_family_block = compute_16_variability(records)
    variability_signal.to_csv(os.path.join(args.output_dir, "16_joint_angle_variability_by_block.csv"), index=False)
    variability_family_block.to_csv(os.path.join(args.output_dir, "16_joint_angle_variability_family_by_block.csv"), index=False)
    variability_family_participant = aggregate_block_features(variability_family_block, skill_col="skill_level")
    variability_family_participant.to_csv(os.path.join(args.output_dir, "16_joint_angle_variability_family_participant.csv"), index=False)

    # Baseline + merge all participant summaries
    print("[4/4] Merging and evaluating candidates...")
    baseline = extract_baseline_participant_features(trial_features, conditions)
    baseline.to_csv(os.path.join(args.output_dir, "baseline_extract_features3_participant.csv"), index=False)

    keys = ["participant_id", "instrument", "condition", "skill_level"]
    master = consistency.copy()
    if "Skill Level" in master.columns:
        master = master.rename(columns={"Skill Level": "skill_level"})
    master = master.merge(pm_participant, on=keys, how="outer", suffixes=("", "_pm"))
    master = master.merge(variability_family_participant, on=keys, how="outer", suffixes=("", "_var"))
    if not baseline.empty:
        master = master.merge(baseline, on=keys, how="outer", suffixes=("", "_baseline"))

    # Harmonize duration columns after merges.
    duration_cols = [c for c in master.columns if c.startswith("mean_duration_s")]
    if duration_cols:
        master["mean_duration_s"] = master[duration_cols].bfill(axis=1).iloc[:, 0]
    master.to_csv(os.path.join(args.output_dir, "unified_pca_participant_features.csv"), index=False)

    pca_candidates = [
        "PCA11b_within_participant_consistency",
        "PCA11b_consistency_sd_pairs",
        "PCA16_lower_body_mean_sd",
        "PCA16_trunk_head_mean_sd",
        "PCA16_upper_body_mean_sd",
        "PCA_n_components_90",
        "PCA_total_variance_explained_by_pc1",
    ]
    pca_candidates += [f"PCA12a_PM{i}_rms" for i in range(1, min(N_PM_TO_SHOW, NB_PM_DETAIL) + 1)]
    pca_candidates = [c for c in pca_candidates if c in master.columns]

    evaluation = skill_duration_evaluation(master, pca_candidates)
    evaluation.to_csv(os.path.join(args.output_dir, "unified_pca_skill_duration_evaluation.csv"), index=False)

    redundancy_candidates = pca_candidates + [c for c in RELATED_FEATURES if c in master.columns]
    redundancy = redundancy_evaluation(master, redundancy_candidates)
    redundancy.to_csv(os.path.join(args.output_dir, "unified_pca_redundancy_correlations.csv"), index=False)

    print("\n=== Shared PM summary (12a defaults) ===")
    print(f"PMs needed for 90% variance: {pm_counts[0.90]}")
    print(f"PMs needed for 95% variance: {pm_counts[0.95]}")
    print(f"Saved PM loadings: all {len(pm_loadings)} PMs")
    print(f"Evaluated PM RMS candidates: PM1-PM{min(N_PM_TO_SHOW, len(pm_loadings))}")

    print("\n=== Candidate feature evaluation ===")
    if evaluation.empty:
        print("No evaluation table could be computed.")
    else:
        display = [
            "instrument", "feature", "duration_pearson_r", "duration_pearson_p",
            "raw_skill_anova_p", "raw_skill_eta_squared", "duration_residual_skill_anova_p",
        ]
        print(evaluation[display].round(4).to_string(index=False))

    high = redundancy[redundancy["flag_high_redundancy_abs_r_ge_0_90"]] if not redundancy.empty else pd.DataFrame()
    print(f"\nHigh-redundancy PCA candidate pairs (|r| >= .90): {len(high)}")
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

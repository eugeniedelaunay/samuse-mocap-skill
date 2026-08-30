#!/usr/bin/env python3
"""
SAMuSe pre-training feature diagnostics.

This script is for feature quality control and exploratory ranking before
training. It does not replace leakage-safe feature selection inside
participant-grouped nested cross-validation.

It reports:
- missingness;
- distributions and near-zero variance;
- IQR/z-score outliers;
- exploratory univariate association with skill;
- feature redundancy (Pearson |r| >= threshold);
- instrument-confound screening;
- optional separate violin and clarinet reports.

Safety defaults:
- excludes metadata, trial duration, status/error, and every QC_* column;
- excludes precomputed *_z columns by default, preventing raw/z duplicates;
- never treats identifier columns as features.

Examples
--------
Raw-feature quality check (default):
python analyze_features2.py \
  --trial_features features_extracted/trial_features.csv \
  --output_dir features_analysis_raw \
  --target_col "Skill Level" \
  --metadata_file metadata/participants_metadata_clean.csv \
  --per_instrument

Normalized-only exploratory view (not final CV evidence):
python analyze_features2.py \
  --trial_features features_extracted/trial_features.csv \
  --output_dir features_analysis_z \
  --target_col "Skill Level" \
  --metadata_file metadata/participants_metadata_clean.csv \
  --per_instrument \
  --feature_set z_only
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

NON_FEATURE_COLS = {
    "participant_id", "condition", "block", "file", "status", "error",
    "instrument", "n_rows", "trial_duration_seconds", "mean_block_duration_seconds",
    "ID", "Instrument", "Skill Level", "Num Level", "Age", "Sex",
    "n_trials", "skill_level", "skill_group",
}

NEAR_ZERO_VAR_THRESHOLD = 1e-8
OUTLIER_IQR_MULTIPLIER = 1.5
OUTLIER_Z_THRESHOLD = 3.0
HIGH_CORR_THRESHOLD = 0.90
MISSINGNESS_WARN_THRESHOLD = 0.30


def load_table(path):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".csv":
        return pd.read_csv(path)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {extension}")


def normalize_participant_id(series):
    return series.astype(str).str.strip().str.upper()


def load_and_merge(trial_features_path, metadata_file=None):
    dataframe = pd.read_csv(trial_features_path)
    if "status" in dataframe.columns:
        dataframe = dataframe[dataframe["status"] == "ok"].copy()

    if metadata_file:
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_file}")
        metadata = load_table(metadata_file)
        metadata.columns = [str(column).strip() for column in metadata.columns]
        id_column = next(
            (column for column in metadata.columns if column.strip().lower() in {"id", "participant_id", "participant"}),
            None,
        )
        if id_column is None:
            raise ValueError(
                "Could not identify participant ID column in metadata. "
                f"Available columns: {list(metadata.columns)}"
            )
        metadata = metadata.rename(columns={id_column: "participant_id"})
        metadata["participant_id"] = normalize_participant_id(metadata["participant_id"])
        dataframe["participant_id"] = normalize_participant_id(dataframe["participant_id"])

        if metadata["participant_id"].duplicated().any():
            duplicated = metadata.loc[metadata["participant_id"].duplicated(keep=False), "participant_id"].unique()
            raise ValueError(
                "Metadata has duplicate participant IDs; merge would duplicate trial rows. "
                f"Examples: {duplicated[:10].tolist()}"
            )
        dataframe = dataframe.merge(metadata, on="participant_id", how="left", suffixes=("", "_meta"), validate="many_to_one")

    return dataframe


def resolve_target_column(dataframe, requested_target):
    if requested_target in dataframe.columns:
        return requested_target
    lookup = {str(column).strip().lower(): column for column in dataframe.columns}
    match = lookup.get(requested_target.strip().lower())
    if match is not None:
        return match
    candidates = [column for column in dataframe.columns if "skill" in str(column).lower()]
    raise ValueError(
        f"Target column '{requested_target}' was not found. "
        f"Skill-like columns available: {candidates}"
    )


def get_feature_columns(dataframe, feature_set="raw_only"):
    columns = []
    for column in dataframe.columns:
        if column in NON_FEATURE_COLS:
            continue
        if str(column).startswith("QC_"):
            continue
        if str(column).endswith("_valid_frames"):
            continue
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            continue

        is_z = str(column).endswith("_z")
        if feature_set == "raw_only" and is_z:
            continue
        if feature_set == "z_only" and not is_z:
            continue
        columns.append(column)
    return columns


def analyze_missingness(dataframe, feature_columns):
    rows = []
    for feature in feature_columns:
        n_total = len(dataframe)
        n_missing = int(dataframe[feature].isna().sum())
        fraction = n_missing / n_total if n_total else np.nan
        rows.append({
            "feature": feature,
            "n_total": n_total,
            "n_missing": n_missing,
            "pct_missing": float(fraction),
            "flag_high_missing": bool(fraction > MISSINGNESS_WARN_THRESHOLD),
        })
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


def analyze_distributions(dataframe, feature_columns):
    rows = []
    for feature in feature_columns:
        values = pd.to_numeric(dataframe[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < 2:
            rows.append({
                "feature": feature,
                "n_valid": len(values),
                "mean": np.nan,
                "sd": np.nan,
                "min": np.nan,
                "max": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "variance": np.nan,
                "flag_near_zero_variance": True,
            })
            continue
        variance = float(np.var(values))
        rows.append({
            "feature": feature,
            "n_valid": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "skewness": float(stats.skew(values)),
            "kurtosis": float(stats.kurtosis(values)),
            "variance": variance,
            "flag_near_zero_variance": bool(variance < NEAR_ZERO_VAR_THRESHOLD),
        })
    return pd.DataFrame(rows).sort_values("variance", ascending=True, na_position="first")


def analyze_outliers(dataframe, feature_columns):
    summary_rows = []
    detail_rows = []
    for feature in feature_columns:
        values = pd.to_numeric(dataframe[feature], errors="coerce").dropna()
        if len(values) < 4:
            continue
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        iqr_mask = (values < lower) | (values > upper)

        standard_deviation = values.std(ddof=1)
        z_scores = (values - values.mean()) / standard_deviation if standard_deviation > 0 else pd.Series(0.0, index=values.index)
        z_mask = z_scores.abs() > OUTLIER_Z_THRESHOLD
        outlier_indices = values.index[iqr_mask | z_mask]

        summary_rows.append({
            "feature": feature,
            "n_valid": len(values),
            "n_outliers_iqr": int(iqr_mask.sum()),
            "pct_outliers_iqr": float(iqr_mask.mean()),
            "n_outliers_zscore": int(z_mask.sum()),
            "pct_outliers_zscore": float(z_mask.mean()),
            "iqr_lower_bound": float(lower),
            "iqr_upper_bound": float(upper),
        })
        for index in outlier_indices:
            detail_rows.append({
                "feature": feature,
                "row_index": int(index),
                "participant_id": dataframe.loc[index, "participant_id"] if "participant_id" in dataframe.columns else None,
                "condition": dataframe.loc[index, "condition"] if "condition" in dataframe.columns else None,
                "block": dataframe.loc[index, "block"] if "block" in dataframe.columns else None,
                "value": float(values.loc[index]),
                "iqr_outlier": bool(iqr_mask.loc[index]),
                "zscore_outlier": bool(z_mask.loc[index]),
            })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("pct_outliers_iqr", ascending=False)
    details = pd.DataFrame(detail_rows)
    return summary, details


def analyze_discriminative_power(dataframe, feature_columns, target_column):
    subset = dataframe[dataframe[target_column].notna()].copy()
    if subset.empty:
        return pd.DataFrame()
    labels = LabelEncoder().fit_transform(subset[target_column].astype(str))
    rows = []

    for feature in feature_columns:
        values = pd.to_numeric(subset[feature], errors="coerce")
        valid = values.notna()
        if valid.sum() < 10 or len(np.unique(labels[valid.to_numpy()])) < 2:
            continue
        x = values.loc[valid].to_numpy(dtype=float)
        y = labels[valid.to_numpy()]
        groups = [x[y == label] for label in np.unique(y)]
        groups = [group for group in groups if len(group) >= 2]
        if len(groups) < 2:
            continue

        try:
            anova_f, anova_p = stats.f_oneway(*groups)
        except Exception:
            anova_f, anova_p = np.nan, np.nan
        try:
            kruskal_h, kruskal_p = stats.kruskal(*groups)
        except Exception:
            kruskal_h, kruskal_p = np.nan, np.nan
        try:
            mutual_information = mutual_info_classif(x.reshape(-1, 1), y, discrete_features=False, random_state=42)[0]
        except Exception:
            mutual_information = np.nan

        rows.append({
            "feature": feature,
            "n_valid": int(valid.sum()),
            "anova_F": float(anova_f) if np.isfinite(anova_f) else np.nan,
            "anova_p": float(anova_p) if np.isfinite(anova_p) else np.nan,
            "kruskal_H": float(kruskal_h) if np.isfinite(kruskal_h) else np.nan,
            "kruskal_p": float(kruskal_p) if np.isfinite(kruskal_p) else np.nan,
            "mutual_info": float(mutual_information) if np.isfinite(mutual_information) else np.nan,
            "significant_anova_p_lt_0_05": bool(np.isfinite(anova_p) and anova_p < 0.05),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["mutual_info", "anova_p"], ascending=[False, True], na_position="last")
    return result


def analyze_multicollinearity(dataframe, feature_columns):
    numeric = dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")
    correlations = numeric.corr(method="pearson")
    pairs = []
    for index, feature_a in enumerate(feature_columns):
        for feature_b in feature_columns[index + 1:]:
            correlation = correlations.loc[feature_a, feature_b]
            if np.isfinite(correlation) and abs(correlation) >= HIGH_CORR_THRESHOLD:
                pairs.append({
                    "feature_1": feature_a,
                    "feature_2": feature_b,
                    "correlation": float(correlation),
                    "absolute_correlation": float(abs(correlation)),
                })
    pairs = pd.DataFrame(pairs)
    if not pairs.empty:
        pairs = pairs.sort_values("absolute_correlation", ascending=False)
    else:
        pairs = pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "absolute_correlation"])
    return correlations, pairs


def analyze_instrument_confound(dataframe, feature_columns, instrument_column="instrument"):
    if instrument_column not in dataframe.columns:
        return pd.DataFrame()
    instruments = dataframe[instrument_column].dropna().unique()
    if len(instruments) < 2:
        return pd.DataFrame()

    rows = []
    for feature in feature_columns:
        groups = []
        means = {}
        for instrument in instruments:
            values = pd.to_numeric(
                dataframe.loc[dataframe[instrument_column] == instrument, feature], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            if len(values) >= 2:
                groups.append(values)
            means[f"mean_{instrument}"] = float(np.mean(values)) if len(values) else np.nan
        if len(groups) < 2:
            continue
        try:
            statistic, p_value = stats.f_oneway(*groups)
        except Exception:
            statistic, p_value = np.nan, np.nan
        rows.append({
            "feature": feature,
            "anova_F_by_instrument": float(statistic) if np.isfinite(statistic) else np.nan,
            "anova_p_by_instrument": float(p_value) if np.isfinite(p_value) else np.nan,
            "differs_significantly_by_instrument": bool(np.isfinite(p_value) and p_value < 0.05),
            **means,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("anova_p_by_instrument", ascending=True)
    return result


def build_final_ranking(missingness, distributions, outliers, discrimination, confounds):
    ranking = distributions[["feature", "n_valid", "variance", "flag_near_zero_variance"]].copy()
    ranking = ranking.merge(
        missingness[["feature", "pct_missing", "flag_high_missing"]], on="feature", how="left"
    )
    if not outliers.empty:
        ranking = ranking.merge(outliers[["feature", "pct_outliers_iqr"]], on="feature", how="left")
    else:
        ranking["pct_outliers_iqr"] = np.nan
    if not discrimination.empty:
        ranking = ranking.merge(
            discrimination[["feature", "anova_F", "anova_p", "mutual_info", "significant_anova_p_lt_0_05"]],
            on="feature",
            how="left",
        )
    else:
        ranking["anova_F"] = np.nan
        ranking["anova_p"] = np.nan
        ranking["mutual_info"] = np.nan
        ranking["significant_anova_p_lt_0_05"] = False
    if not confounds.empty:
        ranking = ranking.merge(
            confounds[["feature", "anova_p_by_instrument", "differs_significantly_by_instrument"]],
            on="feature",
            how="left",
        )
    else:
        ranking["anova_p_by_instrument"] = np.nan
        ranking["differs_significantly_by_instrument"] = False

    def score(row):
        value = 0.0
        if pd.notna(row.get("mutual_info")):
            value += float(row["mutual_info"]) * 10.0
        if bool(row.get("significant_anova_p_lt_0_05", False)):
            value += 1.0
        if bool(row.get("flag_near_zero_variance", False)):
            value -= 5.0
        if bool(row.get("flag_high_missing", False)):
            value -= 3.0
        if pd.notna(row.get("pct_outliers_iqr")) and row["pct_outliers_iqr"] > 0.10:
            value -= 1.0
        if bool(row.get("differs_significantly_by_instrument", False)):
            value -= 0.5
        return value

    ranking["composite_score"] = ranking.apply(score, axis=1)
    ranking = ranking.sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def write_reports(dataframe, feature_columns, target_column, output_dir, prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    missingness = analyze_missingness(dataframe, feature_columns)
    distributions = analyze_distributions(dataframe, feature_columns)
    outlier_summary, outlier_details = analyze_outliers(dataframe, feature_columns)
    discrimination = analyze_discriminative_power(dataframe, feature_columns, target_column)
    correlations, redundant_pairs = analyze_multicollinearity(dataframe, feature_columns)
    confounds = analyze_instrument_confound(dataframe, feature_columns)
    ranking = build_final_ranking(missingness, distributions, outlier_summary, discrimination, confounds)

    missingness.to_csv(os.path.join(output_dir, f"{prefix}missingness.csv"), index=False)
    distributions.to_csv(os.path.join(output_dir, f"{prefix}distributions.csv"), index=False)
    outlier_summary.to_csv(os.path.join(output_dir, f"{prefix}outlier_summary.csv"), index=False)
    outlier_details.to_csv(os.path.join(output_dir, f"{prefix}outlier_details.csv"), index=False)
    discrimination.to_csv(os.path.join(output_dir, f"{prefix}discriminative_power.csv"), index=False)
    correlations.to_csv(os.path.join(output_dir, f"{prefix}feature_correlations.csv") )
    redundant_pairs.to_csv(os.path.join(output_dir, f"{prefix}redundant_feature_pairs.csv"), index=False)
    confounds.to_csv(os.path.join(output_dir, f"{prefix}instrument_confound.csv"), index=False)
    ranking.to_csv(os.path.join(output_dir, f"{prefix}feature_ranking.csv"), index=False)

    return {
        "missingness": missingness,
        "distributions": distributions,
        "outliers": outlier_summary,
        "discrimination": discrimination,
        "redundancy": redundant_pairs,
        "confounds": confounds,
        "ranking": ranking,
    }


def print_summary(reports, title):
    ranking = reports["ranking"]
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Total features analyzed: {len(ranking)}")
    print(f"Features with >30% missing: {int(reports['missingness']['flag_high_missing'].sum())}")
    print(f"Near-zero-variance features: {int(reports['distributions']['flag_near_zero_variance'].sum())}")
    print(f"Redundant |r| >= {HIGH_CORR_THRESHOLD:.2f} pairs: {len(reports['redundancy'])}")
    if not reports["confounds"].empty:
        print(f"Features differing by instrument (p<.05): {int(reports['confounds']['differs_significantly_by_instrument'].sum())}")

    print("\nTop 10 exploratory features:")
    columns = [column for column in ["rank", "feature", "composite_score", "mutual_info", "anova_p"] if column in ranking.columns]
    print(ranking.head(10)[columns].to_string(index=False))
    print("\nBottom 10 candidates to inspect/drop:")
    columns = [column for column in ["rank", "feature", "composite_score", "mutual_info", "flag_near_zero_variance", "flag_high_missing"] if column in ranking.columns]
    print(ranking.tail(10)[columns].to_string(index=False))
    print("\nReminder: these univariate trial-row statistics are exploratory only. "
          "Decide final features inside participant-grouped nested CV.")


def main():
    parser = argparse.ArgumentParser(description="SAMuSe pre-training feature diagnostics with QC and duplicate safeguards.")
    parser.add_argument("--trial_features", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--metadata_file", default=None)
    parser.add_argument("--per_instrument", action="store_true")
    parser.add_argument(
        "--feature_set",
        choices=["raw_only", "z_only", "all"],
        default="raw_only",
        help="raw_only excludes *_z duplicates (default); z_only analyzes only *_z; all is discouraged.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataframe = load_and_merge(args.trial_features, args.metadata_file)
    target_column = resolve_target_column(dataframe, args.target_col)
    feature_columns = get_feature_columns(dataframe, args.feature_set)
    if not feature_columns:
        raise ValueError(f"No numeric feature columns found for feature_set={args.feature_set}.")

    print(f"Loaded {len(dataframe)} successful trial rows, {len(feature_columns)} numeric features ({args.feature_set}).")
    print("QC_* columns, metadata, duration, and raw/z duplicates are excluded according to --feature_set.")
    print("\n[1/7] Analyzing missingness...")
    print("[2/7] Analyzing distributions...")
    print("[3/7] Detecting outliers...")
    print("[4/7] Assessing exploratory discriminative power vs target...")
    print("[5/7] Checking multicollinearity...")
    print("[6/7] Checking instrument confound...")
    print("[7/7] Building final composite ranking...")

    overall_reports = write_reports(dataframe, feature_columns, target_column, args.output_dir, prefix="overall_")
    print_summary(overall_reports, "OVERALL FEATURE-DIAGNOSTIC SUMMARY")

    if args.per_instrument:
        if "instrument" not in dataframe.columns:
            print("\nPer-instrument analysis skipped: no instrument column available.")
        else:
            side_by_side = []
            for instrument in ("violin", "clarinet"):
                subset = dataframe[dataframe["instrument"].astype(str).str.lower() == instrument].copy()
                if subset.empty:
                    print(f"\nNo {instrument} rows found; skipped.")
                    continue
                reports = write_reports(
                    subset,
                    feature_columns,
                    target_column,
                    args.output_dir,
                    prefix=f"{instrument}_",
                )
                print_summary(reports, f"{instrument.upper()} FEATURE-DIAGNOSTIC SUMMARY")
                ranking = reports["ranking"][["feature", "rank", "composite_score", "mutual_info", "anova_p"]].copy()
                ranking = ranking.rename(columns={
                    "rank": f"{instrument}_rank",
                    "composite_score": f"{instrument}_composite_score",
                    "mutual_info": f"{instrument}_mutual_info",
                    "anova_p": f"{instrument}_anova_p",
                })
                side_by_side.append(ranking)
            if side_by_side:
                comparison = side_by_side[0]
                for table in side_by_side[1:]:
                    comparison = comparison.merge(table, on="feature", how="outer")
                comparison.to_csv(os.path.join(args.output_dir, "per_instrument_feature_comparison.csv"), index=False)

    print(f"\nAll reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

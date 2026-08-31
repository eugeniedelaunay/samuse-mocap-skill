"""
merge_skill_labels.py -- SAMuSe skill-label merge stage.

Merges the skill/demographic metadata (participant-level: ID, Instrument,
Skill Level, Num Level, Age, Sex) into the trial-level feature table
produced by extract_features3.py, producing trial_features_with_skill.csv.

Why this exists
----------------
extract_features3.py computes ROM/SPARC/LDJ/PCA features per trial but does
NOT merge in the participant's skill label -- that lives only in the
metadata spreadsheet. train_skill_classifiers.py, train_skill_classifiers_grouped.py,
and analyze_features2.py all require a "Skill Level" column to already be
present in --trial_features, so this merge must run between extract_features
and those training/analysis stages.

Metadata file quirks handled here
----------------------------------
The metadata .xlsx has extra non-data columns (a small summary/pivot table
living in the same sheet, e.g. "violin", "Total", "Unnamed: 8", bare integers
like 19/7/1/2/3/4) alongside the six real participant columns:
ID, Instrument, Skill Level, Num Level, Age, Sex.
This script explicitly selects only those six columns and ignores the rest.

Not every participant in the metadata file necessarily has usable trial
rows (e.g. excluded/failed trials) -- this script performs an inner-style
join and reports exactly which participants were dropped and why, rather
than silently losing them.

Usage
-----
python src/merge_skill_labels.py \
  --trial_features data/features_extracted/trial_features.csv \
  --metadata_file data/metadata/participants_instrument_skill_toshare.xlsx \
  --output_file data/features_extracted/trial_features_with_skill.csv
"""

import argparse
import os
from pathlib import Path

import pandas as pd

REQUIRED_METADATA_COLUMNS = ["ID", "Instrument", "Skill Level", "Num Level", "Age", "Sex"]


def load_metadata(metadata_file: str) -> pd.DataFrame:
    extension = Path(metadata_file).suffix.lower()
    if extension == ".csv":
        metadata = pd.read_csv(metadata_file)
    elif extension in {".xlsx", ".xls"}:
        # Explicitly select only the known real columns; the sheet also
        # contains an unrelated summary/pivot block that read_excel would
        # otherwise pull in as extra junk columns.
        metadata = pd.read_excel(metadata_file, usecols=REQUIRED_METADATA_COLUMNS)
    else:
        raise ValueError(f"Unsupported metadata extension: {extension}")

    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(
            f"Metadata file is missing required column(s): {missing}. "
            f"Found: {list(metadata.columns)}"
        )

    metadata = metadata[REQUIRED_METADATA_COLUMNS].copy()
    metadata["ID"] = metadata["ID"].astype(str).str.strip().str.upper()

    duplicated = metadata["ID"].duplicated()
    if duplicated.any():
        duplicate_ids = metadata.loc[duplicated, "ID"].unique().tolist()
        raise ValueError(f"Metadata has duplicate participant ID(s): {duplicate_ids}")

    return metadata.rename(columns={"ID": "participant_id"})


def load_trial_features(trial_features_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(trial_features_path)
    if "participant_id" not in dataframe.columns:
        raise ValueError(
            f"{trial_features_path} has no 'participant_id' column; "
            "cannot merge skill labels."
        )
    dataframe["participant_id"] = dataframe["participant_id"].astype(str).str.strip().str.upper()
    return dataframe


def merge_skill_labels(trial_features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    feature_participants = set(trial_features["participant_id"].unique())
    metadata_participants = set(metadata["participant_id"].unique())

    unmatched_in_features = sorted(feature_participants - metadata_participants)
    unmatched_in_metadata = sorted(metadata_participants - feature_participants)

    if unmatched_in_features:
        print(
            f"WARNING: {len(unmatched_in_features)} participant(s) in trial_features "
            f"have NO matching metadata row and will have missing skill labels: "
            f"{unmatched_in_features}"
        )
    if unmatched_in_metadata:
        print(
            f"INFO: {len(unmatched_in_metadata)} participant(s) in metadata have NO "
            f"matching trial_features rows and are naturally excluded from output: "
            f"{unmatched_in_metadata}"
        )

    merged = trial_features.merge(
        metadata,
        on="participant_id",
        how="left",
        validate="many_to_one",
        indicator="_metadata_merge",
    )

    missing_label_rows = merged["_metadata_merge"] != "both"
    if missing_label_rows.any():
        print(
            f"WARNING: {int(missing_label_rows.sum())} trial row(s) have no skill label "
            f"after merge (participant not found in metadata)."
        )

    merged = merged.drop(columns="_metadata_merge")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge participant skill/demographic metadata into trial-level features."
    )
    parser.add_argument("--trial_features", required=True, help="Path to trial_features.csv (output of extract_features3.py)")
    parser.add_argument("--metadata_file", required=True, help="Path to participant metadata CSV/XLSX")
    parser.add_argument("--output_file", required=True, help="Path to write trial_features_with_skill.csv")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata_file)
    trial_features = load_trial_features(args.trial_features)

    print(f"Loaded {len(trial_features)} trial rows across {trial_features['participant_id'].nunique()} participants from trial_features.")
    print(f"Loaded {len(metadata)} participants from metadata.")
    print("Metadata Skill Level distribution:")
    print(metadata["Skill Level"].value_counts().to_string())

    merged = merge_skill_labels(trial_features, metadata)

    n_participants_final = merged.loc[merged["Skill Level"].notna(), "participant_id"].nunique()
    print(f"\nFinal merged table: {len(merged)} trial rows, {n_participants_final} participants with a valid skill label.")
    print("Final Skill Level distribution (trial rows):")
    print(merged["Skill Level"].value_counts(dropna=False).to_string())

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    merged.to_csv(args.output_file, index=False)
    print(f"\nSaved: {args.output_file}")


if __name__ == "__main__":
    main()
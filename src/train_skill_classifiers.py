"""
NOTE (corrections):
- ROM_rate_* (ROM divided by trial duration) is excluded from the baseline set.
- In pooled mode, any feature with fewer than 2 valid values for any
  instrument (e.g. left-wrist stroke LDJ, PM4, PM6, windowed SPARC) is
  dropped, so that imputation cannot reveal the instrument.
- Logistic regression C = 0.5, linear SVM C = 0.1 (LOGREG_C, SVM_C).

SAMuSe trial-wise skill-classification experiments.

Teacher-specified evaluation protocol
-------------------------------------
- Trial-wise StratifiedKFold: trials are split independently.
- Different trials from the same participant may occur in train and test folds.
- This measures held-out TRIAL performance, not generalization to unseen
  participants.
- Competent is merged into advanced_beginner for 3-class classification.
- Balanced accuracy is the primary score.

Feature processing is fit inside each training fold only:
median imputation -> variance filtering -> correlation pruning -> scaling ->
optional SelectKBest -> classifier.

Models
------
- Logistic regression (default)
- Linear SVM
- Random forest

Feature configurations
----------------------
baseline:
    Raw ROM / ROM_rate and LDJ stroke mean/SD features.
shared_pca:
    baseline + six shared PCA candidates.
instrument_specific:
    shared_pca + relevant instrument-specific smoothness/PCA features.
    In pooled mode, instrument-specific feature columns are safely excluded
    because they are absent for the other instrument and would encode
    instrument identity through missingness.

Examples
--------
Pooled, 3-class, shared PCA, logistic regression:
python train_skill_classifiers.py \
  --trial_features features_extracted/trial_features.csv \
  --output_dir training_results/pooled_3class_shared_pca_logreg \
  --task 3class --instrument_mode pooled --feature_config shared_pca \
  --model logreg

Separate instrument, 2-class, instrument-specific features, all models:
python train_skill_classifiers.py \
  --trial_features features_extracted/trial_features.csv \
  --output_dir training_results/separate_2class_instrument_specific \
  --task 2class --instrument_mode separate --feature_config instrument_specific \
  --model all
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

RANDOM_STATE = 42
DEFAULT_SPLITS = 5
HIGH_CORRELATION = 0.90
LOGREG_C = float(os.environ.get("SAMUSE_LOGREG_C", 0.5))
SVM_C = float(os.environ.get("SAMUSE_SVM_C", 0.1))

DROP_EXACT = {
    "participant_id", "condition", "block", "instrument", "file", "status", "error",
    "n_rows", "trial_duration_seconds", "mean_block_duration_seconds", "n_trials",
    "Skill Level", "skill_level", "skill_group", "Num Level", "Age", "Sex",
    "ID", "Instrument",
}

KNOWN_NEAR_ZERO_FEATURES = {
    "ROM_RElbowAngles_X",
    "ROM_RElbowAngles_Z",
    "ROM_LElbowAngles_X",
    "ROM_LElbowAngles_Z",
    "ROM_rate_RElbowAngles_X",
    "ROM_rate_RElbowAngles_Z",
    "ROM_rate_LElbowAngles_X",
    "ROM_rate_LElbowAngles_Z",
    "ROM_rate_Spine4Angles_X",
    "ROM_Spine4Angles_Y",
}

SHARED_PCA_FEATURES = {
    "PCA16_lower_body_mean_sd",
    "PCA16_trunk_head_mean_sd",
    "PCA16_upper_body_mean_sd",
    "PCA12a_PM1_rms",
    "PCA12a_PM2_rms",
    "PCA12a_PM7_rms",
}

CLARINET_SPECIFIC_FEATURES = {
    "PCA11b_within_participant_consistency",
    "PCA12a_PM4_rms",
    "SPARC_window_median_LWrist",
    "LDJ_stroke_mean_LWrist",
    "LDJ_stroke_sd_LWrist",
}

VIOLIN_SPECIFIC_FEATURES = {
    "PCA12a_PM6_rms",
    "SPARC_window_bow_elbow_all_median",
}


def canonical_label(value):
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "advancedbeginner": "advanced_beginner",
        "advanced_beginner": "advanced_beginner",
        "beginner": "novice",
        "novice": "novice",
        "competent": "competent",
        "expert": "expert",
        "proficient_and_expert": "expert",
        "proficient_expert": "expert",
    }
    return aliases.get(text, text)


def find_target_column(dataframe, requested):
    if requested in dataframe.columns:
        return requested
    normalized = {str(column).strip().lower(): column for column in dataframe.columns}
    if requested.strip().lower() in normalized:
        return normalized[requested.strip().lower()]
    candidates = [column for column in dataframe.columns if "skill" in str(column).lower()]
    raise ValueError(f"Target column '{requested}' not found. Skill-like columns: {candidates}")


def load_data(path, target_column, task):
    dataframe = pd.read_csv(path)
    if "status" in dataframe.columns:
        dataframe = dataframe[dataframe["status"] == "ok"].copy()
    target_column = find_target_column(dataframe, target_column)
    dataframe["_label_raw"] = dataframe[target_column].map(canonical_label)

    if task == "3class":
        label_map = {
            "novice": "novice",
            "advanced_beginner": "advanced_beginner",
            "competent": "advanced_beginner",
            "expert": "expert",
        }
        dataframe["_label"] = dataframe["_label_raw"].map(label_map)
        class_order = ["novice", "advanced_beginner", "expert"]
    else:
        dataframe = dataframe[dataframe["_label_raw"].isin(["novice", "expert"])].copy()
        dataframe["_label"] = dataframe["_label_raw"]
        class_order = ["novice", "expert"]

    dataframe = dataframe.dropna(subset=["_label", "instrument"]).copy()
    dataframe["instrument"] = dataframe["instrument"].astype(str).str.strip().str.lower()
    return dataframe, class_order


def candidate_feature_columns(dataframe, config, instrument_mode):
    columns = []
    for column in dataframe.columns:
        if column in DROP_EXACT:
            continue
        if column.startswith("QC_") or column.endswith("_z"):
            continue
        if column in KNOWN_NEAR_ZERO_FEATURES:
            continue
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            continue
        columns.append(column)

    def belongs_to_baseline(column):
        return column.startswith(("ROM_", "LDJ_stroke_")) and not column.startswith("ROM_rate_")

    baseline = [column for column in columns if belongs_to_baseline(column)]
    shared = [column for column in columns if column in SHARED_PCA_FEATURES]

    if config == "baseline":
        selected = baseline
    elif config == "shared_pca":
        selected = baseline + shared
    elif config == "instrument_specific":
        selected = baseline + shared
        if instrument_mode == "separate":
            # Relevant additions are selected later after the data are filtered
            # to a concrete instrument.
            selected += [column for column in columns if column in CLARINET_SPECIFIC_FEATURES | VIOLIN_SPECIFIC_FEATURES]
    else:
        raise ValueError(f"Unknown feature configuration: {config}")

    return sorted(set(selected))


def select_features_for_instrument(feature_columns, instrument, config):
    if config != "instrument_specific":
        return feature_columns
    if instrument == "clarinet":
        allowed = SHARED_PCA_FEATURES | CLARINET_SPECIFIC_FEATURES
    elif instrument == "violin":
        allowed = SHARED_PCA_FEATURES | VIOLIN_SPECIFIC_FEATURES
    else:
        allowed = SHARED_PCA_FEATURES

    selected = []
    for column in feature_columns:
        if column.startswith(("ROM_", "LDJ_stroke_")) or column in allowed:
            selected.append(column)
    return selected


def prune_correlated_train_features(x_train, feature_names, threshold=HIGH_CORRELATION):
    """Fit correlation pruning on training data only; keep first feature in order."""
    train = pd.DataFrame(x_train, columns=feature_names)
    correlations = train.corr(method="pearson").abs()
    to_drop = set()
    for i, left in enumerate(feature_names):
        if left in to_drop:
            continue
        for right in feature_names[i + 1:]:
            if right in to_drop:
                continue
            value = correlations.loc[left, right]
            if np.isfinite(value) and value >= threshold:
                to_drop.add(right)
    keep_indices = [index for index, name in enumerate(feature_names) if name not in to_drop]
    return keep_indices, sorted(to_drop)


def make_classifier(model_name):
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=5000,
	    C=LOGREG_C,
            class_weight="balanced",
            solver="saga",
            random_state=RANDOM_STATE,
        )
    if model_name == "linear_svm":
        return SVC(
            kernel="linear",
	    C=SVM_C,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=5,
            max_depth=8,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name, k_best):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]
    if model_name != "random_forest":
        steps.append(("scaler", StandardScaler()))
    if k_best and k_best > 0:
        steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=k_best)))
    steps.append(("classifier", make_classifier(model_name)))
    return Pipeline(steps)


def fold_selected_feature_names(pipeline, feature_names):
    names = np.asarray(feature_names, dtype=object)
    variance = pipeline.named_steps.get("variance")
    if variance is not None:
        names = names[variance.get_support()]
    selector = pipeline.named_steps.get("select")
    if selector is not None:
        names = names[selector.get_support()]
    return names.tolist()


def run_experiment(dataframe, feature_columns, class_order, model_name, n_splits, k_best, output_dir, run_name):
    x = dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = dataframe["_label"].astype(str).to_numpy()
    sample_info = dataframe[[column for column in ["participant_id", "condition", "block", "instrument", "file"] if column in dataframe.columns]].copy()

    counts = pd.Series(y).value_counts()
    min_class_count = int(counts.min())
    if min_class_count < 2:
        raise ValueError(f"{run_name}: fewer than 2 trials in a class: {counts.to_dict()}")
    n_splits = min(n_splits, min_class_count)
    if n_splits < 2:
        raise ValueError(f"{run_name}: insufficient class counts for cross-validation: {counts.to_dict()}")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    prediction_rows = []
    feature_selection_rows = []
    confusion_total = np.zeros((len(class_order), len(class_order)), dtype=int)

    for fold, (train_index, test_index) in enumerate(splitter.split(x, y), start=1):
        x_train_raw = x.iloc[train_index].copy()
        x_test_raw = x.iloc[test_index].copy()
        y_train, y_test = y[train_index], y[test_index]

        # Median-impute only training data temporarily for correlation pruning.
        correlation_imputer = SimpleImputer(strategy="median")
        x_train_for_correlation = correlation_imputer.fit_transform(x_train_raw)
        keep_indices, dropped_correlated = prune_correlated_train_features(
            x_train_for_correlation,
            feature_columns,
        )
        selected_columns = [feature_columns[index] for index in keep_indices]
        x_train = x_train_raw[selected_columns]
        x_test = x_test_raw[selected_columns]

        effective_k = min(k_best, len(selected_columns)) if k_best and k_best > 0 else None
        pipeline = build_pipeline(model_name, effective_k)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        train_predictions = pipeline.predict(x_train)
        train_balanced_accuracy = balanced_accuracy_score(y_train, train_predictions)

        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_test,
            predictions,
            labels=class_order,
            zero_division=0,
        )
        confusion_total += confusion_matrix(y_test, predictions, labels=class_order)

        fold_rows.append({
            "run_name": run_name,
            "model": model_name,
            "fold": fold,
            "n_train": len(train_index),
            "n_test": len(test_index),
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "train_balanced_accuracy": train_balanced_accuracy,
            "balanced_accuracy": balanced_accuracy,
            "train_test_gap": train_balanced_accuracy - balanced_accuracy,
            "n_features_before_correlation_pruning": len(feature_columns),
            "n_features_after_correlation_pruning": len(selected_columns),
            "n_features_final": len(fold_selected_feature_names(pipeline, selected_columns)),
            "n_correlated_features_removed": len(dropped_correlated),
            **{f"recall_{label}": value for label, value in zip(class_order, recall)},
            **{f"precision_{label}": value for label, value in zip(class_order, precision)},
            **{f"f1_{label}": value for label, value in zip(class_order, fscore)},
            **{f"support_{label}": value for label, value in zip(class_order, support)},
        })

        selected_final = fold_selected_feature_names(pipeline, selected_columns)
        for feature in selected_final:
            feature_selection_rows.append({"run_name": run_name, "model": model_name, "fold": fold, "feature": feature})

        prediction = sample_info.iloc[test_index].copy()
        prediction.insert(0, "fold", fold)
        prediction["true_label"] = y_test
        prediction["predicted_label"] = predictions
        prediction_rows.append(prediction)

    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    selected_features = pd.DataFrame(feature_selection_rows)
    selected_frequency = (
        selected_features.groupby("feature").size().rename("n_folds_selected").reset_index()
        .assign(selection_frequency=lambda table: table["n_folds_selected"] / n_splits)
        .sort_values(["selection_frequency", "feature"], ascending=[False, True])
        if not selected_features.empty else pd.DataFrame(columns=["feature", "n_folds_selected", "selection_frequency"])
    )

    summary = {
        "run_name": run_name,
        "model": model_name,
        "n_trials": len(dataframe),
        "n_features_input": len(feature_columns),
        "n_splits": n_splits,
        "class_counts": json.dumps(counts.to_dict()),
        "balanced_accuracy_mean": float(fold_results["balanced_accuracy"].mean()),
        "balanced_accuracy_sd": float(fold_results["balanced_accuracy"].std(ddof=1)) if n_splits > 1 else 0.0,
        "macro_f1_mean": float(fold_results["macro_f1"].mean()),
        "macro_f1_sd": float(fold_results["macro_f1"].std(ddof=1)) if n_splits > 1 else 0.0,
        "weighted_f1_mean": float(fold_results["weighted_f1"].mean()),
        "weighted_f1_sd": float(fold_results["weighted_f1"].std(ddof=1)) if n_splits > 1 else 0.0,
    }

    run_dir = os.path.join(output_dir, run_name, model_name)
    os.makedirs(run_dir, exist_ok=True)
    fold_results.to_csv(os.path.join(run_dir, "fold_metrics.csv"), index=False)
    predictions.to_csv(os.path.join(run_dir, "trial_predictions.csv"), index=False)
    selected_frequency.to_csv(os.path.join(run_dir, "selected_feature_frequency.csv"), index=False)
    pd.DataFrame(confusion_total, index=class_order, columns=class_order).to_csv(
        os.path.join(run_dir, "confusion_matrix_total.csv")
    )
    with open(os.path.join(run_dir, "experiment_settings.json"), "w", encoding="utf-8") as file:
        json.dump({
            "run_name": run_name,
            "model": model_name,
            "protocol": "trial-wise StratifiedKFold; same participant may occur in train and test folds",
            "class_order": class_order,
            "feature_columns_input": feature_columns,
            "n_splits": n_splits,
            "k_best": k_best,
            "correlation_threshold": HIGH_CORRELATION,
            "random_state": RANDOM_STATE,
            "logreg_C": LOGREG_C,
            "svm_C": SVM_C,
        }, file, indent=2)

    return summary


def drop_instrument_missing_features(dataframe, feature_columns, min_valid=2):
    """Drop features that are (almost) entirely missing for any instrument.

    In pooled models such columns are median-imputed with a constant for that
    instrument, so the imputed constant reveals the instrument.
    """
    kept, dropped = [], []
    for column in feature_columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        valid = values.notna().groupby(dataframe["instrument"]).sum()
        if (valid < min_valid).any():
            dropped.append(column)
        else:
            kept.append(column)
    if dropped:
        print(f"Pooled mode: dropped instrument-missing features (would encode instrument): {dropped}")
    return kept


def run_mode(dataframe, class_order, feature_columns, instrument_mode, config, models, n_splits, k_best, output_dir):
    summaries = []
    if instrument_mode == "pooled":
        # Do not allow feature columns that are intentionally unavailable in
        # the other instrument. Their missingness would encode instrument ID.
        if config == "instrument_specific":
            print("Pooled + instrument_specific requested: using shared_pca feature set to avoid instrument-missingness leakage.")
            config = "shared_pca"
            feature_columns = candidate_feature_columns(dataframe, config, "pooled")
        feature_columns = drop_instrument_missing_features(dataframe, feature_columns)
        run_name = f"pooled_{config}"
        for model_name in models:
            summaries.append(run_experiment(
                dataframe, feature_columns, class_order, model_name,
                n_splits, k_best, output_dir, run_name,
            ))
        return summaries

    for instrument in ("violin", "clarinet"):
        subset = dataframe[dataframe["instrument"] == instrument].copy()
        if subset.empty:
            print(f"No {instrument} trials; skipped.")
            continue
        instrument_features = select_features_for_instrument(feature_columns, instrument, config)
        # Keep only columns that have at least two non-missing values in the chosen data.
        instrument_features = [
            column for column in instrument_features
            if column in subset.columns and pd.to_numeric(subset[column], errors="coerce").notna().sum() >= 2
        ]
        if not instrument_features:
            print(f"{instrument}: no usable features after filtering; skipped.")
            continue
        run_name = f"{instrument}_{config}"
        for model_name in models:
            summaries.append(run_experiment(
                subset, instrument_features, class_order, model_name,
                n_splits, k_best, output_dir, run_name,
            ))
    return summaries


def main():
    parser = argparse.ArgumentParser(description="SAMuSe trial-wise balanced-accuracy training experiments.")
    parser.add_argument("--trial_features", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_col", default="Skill Level")
    parser.add_argument("--task", choices=["3class", "2class"], required=True)
    parser.add_argument("--instrument_mode", choices=["pooled", "separate"], required=True)
    parser.add_argument("--feature_config", choices=["baseline", "shared_pca", "instrument_specific"], required=True)
    parser.add_argument("--model", choices=["logreg", "linear_svm", "random_forest", "all"], default="logreg")
    parser.add_argument("--n_splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--k_best", type=int, default=0,
        help="Optional SelectKBest count fitted inside each training fold; 0 keeps all post-pruning features.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataframe, class_order = load_data(args.trial_features, args.target_col, args.task)
    feature_columns = candidate_feature_columns(dataframe, args.feature_config, args.instrument_mode)
    if not feature_columns:
        raise ValueError("No model features available after applying exclusions and feature configuration.")

    models = ["logreg", "linear_svm", "random_forest"] if args.model == "all" else [args.model]
    print("\nTrial-wise protocol: StratifiedKFold. The same participant may appear in train and test folds.")
    print(f"Task: {args.task}; classes: {class_order}")
    print(f"Instrument mode: {args.instrument_mode}; feature configuration: {args.feature_config}")
    print(f"Rows: {len(dataframe)}; candidate input features: {len(feature_columns)}")
    print("Class counts:", dataframe["_label"].value_counts().to_dict())

    summaries = run_mode(
        dataframe=dataframe,
        class_order=class_order,
        feature_columns=feature_columns,
        instrument_mode=args.instrument_mode,
        config=args.feature_config,
        models=models,
        n_splits=args.n_splits,
        k_best=args.k_best,
        output_dir=args.output_dir,
    )
    summary = pd.DataFrame(summaries).sort_values(["run_name", "balanced_accuracy_mean"], ascending=[True, False])
    summary.to_csv(os.path.join(args.output_dir, "experiment_summary.csv"), index=False)
    print("\n" + "=" * 88)
    print("EXPERIMENT SUMMARY — trial-wise stratified CV")
    print("=" * 88)
    print(summary[[
        "run_name", "model", "n_trials", "n_features_input", "n_splits",
        "balanced_accuracy_mean", "balanced_accuracy_sd", "macro_f1_mean", "macro_f1_sd",
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved all outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

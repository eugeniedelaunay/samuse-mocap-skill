# SAMuSe MoCap Skill Pipeline

Preprocessing, feature extraction, unified PCA evaluation, feature diagnostics,
and skill-level classification for violin and clarinet motion-capture data.

## Structure

- src/ — pipeline scripts
  - preprocess.py — Stage 1-3 MoCap preprocessing (clean, gap-fill, filter)
  - unified_pca_evaluation.py — PCA / PM / movement-variability feature evaluation
  - extract_features3.py — trial-level feature extraction (ROM, SPARC, LDJ, PCA merge)
  - nalyze_features2.py — feature diagnostics and QC
  - 	rain_skill_classifiers.py — trial-wise skill classification (StratifiedKFold)
  - 	rain_skill_classifiers_grouped.py — participant-grouped skill classification (StratifiedGroupKFold)

- data/
  - metadata/participants_instrument_skill_toshare.xlsx — participant instrument/skill metadata
  - pca/unified_pca_participant_features.csv — unified PCA participant-level features
  - eatures_extracted/trial_features.csv — trial-level extracted features (with skill labels)

## Setup

```bash
pip install -r requirements.txt
```

## Example commands

Feature diagnostics:
```bash
python src/analyze_features2.py \
  --trial_features data/features_extracted/trial_features.csv \
  --output_dir features_analysis_raw \
  --target_col "Skill Level" \
  --metadata_file data/metadata/participants_instrument_skill_toshare.xlsx \
  --per_instrument
```

Trial-wise skill classification:
```bash
python src/train_skill_classifiers.py \
  --trial_features data/features_extracted/trial_features.csv \
  --output_dir training_results/pooled_3class_shared_pca_logreg \
  --task 3class --instrument_mode pooled --feature_config shared_pca \
  --model logreg
```

Participant-grouped skill classification:
```bash
python src/train_skill_classifiers_grouped.py \
  --trial_features data/features_extracted/trial_features.csv \
  --output_dir training_results_grouped/pooled_3class_shared_pca \
  --task 3class --instrument_mode pooled --feature_config shared_pca --model all
```

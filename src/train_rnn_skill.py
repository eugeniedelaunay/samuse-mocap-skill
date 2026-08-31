"""
SAMuSe LSTM/GRU skill-classification from raw joint-position sequences.

Loads cleaned MoCap trial CSVs (as produced by preprocess.py), builds
variable-length sequences of selected joint positions, and trains an
LSTM or GRU classifier to predict skill level.

Evaluation protocols
---------------------
- trial: StratifiedKFold on trials (same participant may appear in
  train and test folds).
- participant: StratifiedGroupKFold on participant_id (no participant
  appears in both train and test folds).

Within each fold, a portion of the training trials is held out as a
validation set for early stopping; the fold's test set is only used
for final evaluation.

Example:
python train_rnn_skill.py \
  --input_dir preprocessed/ \
  --metadata_file metadata/participants_instrument_skill_toshare.xlsx \
  --output_dir training_results_rnn/clarinet_2class_gru \
  --instrument clarinet \
  --task 2class \
  --protocol participant \
  --rnn_type gru \
  --n_splits 5 \
  --epochs 60
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path
import random


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
    GroupShuffleSplit,
)
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

RANDOM_STATE = 42
FRAME_RATE = 50.0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Joint position columns used as RNN input channels.
POSITION_COLUMNS = [
    "RWristPositions_X (mm)", "RWristPositions_Y (mm)", "RWristPositions_Z (mm)",
    "RElbowPositions_X (mm)", "RElbowPositions_Y (mm)", "RElbowPositions_Z (mm)",
    "RShoulderPositions_X (mm)", "RShoulderPositions_Y (mm)", "RShoulderPositions_Z (mm)",
    "LWristPositions_X (mm)", "LWristPositions_Y (mm)", "LWristPositions_Z (mm)",
    "LElbowPositions_X (mm)", "LElbowPositions_Y (mm)", "LElbowPositions_Z (mm)",
    "LShoulderPositions_X (mm)", "LShoulderPositions_Y (mm)", "LShoulderPositions_Z (mm)",
]


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


def load_metadata(metadata_file):
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
    skill_column = next((column for column in metadata.columns if "skill" in column), None)
    if id_column is None or instrument_column is None or skill_column is None:
        raise ValueError(
            "Metadata needs participant-ID, instrument and skill columns. "
            f"Found: {list(metadata.columns)}"
        )

    mapping = {}
    for _, row in metadata.iterrows():
        participant_id = str(row[id_column]).strip().upper()
        instrument = str(row[instrument_column]).strip().lower()
        skill_raw = canonical_label(row[skill_column])
        mapping[participant_id] = {"instrument": instrument, "skill_raw": skill_raw}
    return mapping


def parse_trial_metadata(filepath):
    stem = Path(filepath).stem.replace("_clean", "")
    match = re.match(r"(P\d+)_([a-zA-Z]+)_(\d+)", stem)
    if match is None:
        raise ValueError(f"Cannot parse P###_condition_block from: {Path(filepath).name}")
    participant_id, condition, block = match.groups()
    return participant_id.upper(), condition.lower(), int(block)


def map_task_label(label_raw, task):
    if task == "3class":
        label_map = {
            "novice": "novice",
            "advanced_beginner": "advanced_beginner",
            "competent": "advanced_beginner",
            "expert": "expert",
        }
        return label_map.get(label_raw)
    if label_raw in ("novice", "expert"):
        return label_raw
    return None


def collect_trials(input_dir, metadata_map, instrument_filter, task, conditions, pattern="*_clean.csv"):
    files = sorted(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
    rows = []
    for filepath in files:
        try:
            participant_id, condition, block = parse_trial_metadata(filepath)
        except ValueError:
            continue
        if conditions is not None and condition not in conditions:
            continue
        info = metadata_map.get(participant_id)
        if info is None:
            continue
        if instrument_filter is not None and info["instrument"] != instrument_filter:
            continue
        label = map_task_label(info["skill_raw"], task)
        if label is None:
            continue
        rows.append({
            "file": filepath,
            "participant_id": participant_id,
            "condition": condition,
            "block": block,
            "instrument": info["instrument"],
            "label": label,
        })
    if not rows:
        raise ValueError("No usable trials found. Check input_dir, metadata_file and instrument/task filters.")
    return pd.DataFrame(rows)


def load_trial_sequence(filepath, position_columns):
    dataframe = pd.read_csv(filepath)
    available = [column for column in position_columns if column in dataframe.columns]
    if len(available) < 3:
        return None
    values = dataframe[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if len(values) < 8:
        return None
    values = pd.DataFrame(values).interpolate(limit=5, limit_direction="both").to_numpy(dtype=np.float32)
    if np.isnan(values).any():
        col_mean = np.nanmean(values, axis=0)
        col_mean = np.nan_to_num(col_mean, nan=0.0)
        nan_mask = np.isnan(values)
        values[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return values


class TrialSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return torch.from_numpy(sequence).float(), len(sequence), int(label)


def collate_batch(batch):
    batch = sorted(batch, key=lambda item: item[1], reverse=True)
    sequences, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    max_len = lengths.max().item()
    feature_dim = sequences[0].shape[1]
    padded = torch.zeros(len(sequences), max_len, feature_dim, dtype=torch.float32)
    for index, sequence in enumerate(sequences):
        padded[index, :sequence.shape[0], :] = sequence
    return padded, lengths, labels


class SkillRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, rnn_type="gru", dropout=0.3):
        super().__init__()
        rnn_class = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, hidden = self.rnn(packed)
        last_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        final_layer = last_hidden[-1]
        return self.fc(self.dropout(final_layer))


def normalize_sequences(sequences, scaler=None):
    if scaler is None:
        stacked = np.concatenate(sequences, axis=0)
        scaler = StandardScaler().fit(stacked)
    normalized = [scaler.transform(sequence).astype(np.float32) for sequence in sequences]
    return normalized, scaler


def run_epoch(model, loader, criterion, optimizer, device, train_mode):
    model.train() if train_mode else model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for sequences, lengths, labels in loader:
            sequences, lengths, labels = sequences.to(device), lengths, labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(sequences, lengths)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
            all_pred.extend(predictions.tolist())
            all_true.extend(labels.detach().cpu().numpy().tolist())

    mean_loss = total_loss / len(all_true)
    balanced_accuracy = balanced_accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return mean_loss, balanced_accuracy, macro_f1, all_true, all_pred


def train_one_fold(
    train_sequences, train_labels,
    test_sequences, test_labels,
    class_order, args, device, fold, run_dir,
    train_participants=None,
):
    label_to_index = {label: index for index, label in enumerate(class_order)}
    y_train_full = np.array([label_to_index[label] for label in train_labels])
    y_test = np.array([label_to_index[label] for label in test_labels])

    if args.protocol == "participant" and train_participants is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, val_idx = next(splitter.split(
            np.arange(len(train_sequences)),
            y_train_full,
            groups=train_participants,
        ))
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_sequences)),
            test_size=0.2,
            stratify=y_train_full,
            random_state=RANDOM_STATE,
        )

    train_sequences_fold = [train_sequences[i] for i in train_idx]
    val_sequences_fold = [train_sequences[i] for i in val_idx]
    y_train = y_train_full[train_idx]
    y_val = y_train_full[val_idx]

    train_sequences_fold, scaler = normalize_sequences(train_sequences_fold)
    val_sequences_fold, _ = normalize_sequences(val_sequences_fold, scaler=scaler)
    test_sequences_norm, _ = normalize_sequences(test_sequences, scaler=scaler)

    train_dataset = TrialSequenceDataset(train_sequences_fold, y_train)
    val_dataset = TrialSequenceDataset(val_sequences_fold, y_val)
    test_dataset = TrialSequenceDataset(test_sequences_norm, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    class_counts = np.bincount(y_train, minlength=len(class_order))
    class_weights = torch.tensor(
        [len(y_train) / (len(class_order) * max(count, 1)) for count in class_counts],
        dtype=torch.float32,
    ).to(device)

    model = SkillRNN(
        input_dim=train_sequences_fold[0].shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(class_order),
        rnn_type=args.rnn_type,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_balanced_accuracy = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_balanced_accuracy, train_macro_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train_mode=True,
        )
        val_loss, val_balanced_accuracy, val_macro_f1, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, train_mode=False,
        )
        print(
            f"Fold {fold} | epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_bal_acc={train_balanced_accuracy:.3f} | "
            f"val_loss={val_loss:.4f} val_bal_acc={val_balanced_accuracy:.3f}"
        )

        if val_balanced_accuracy > best_val_balanced_accuracy:
            best_val_balanced_accuracy = val_balanced_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Fold {fold}: early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train_loss, final_train_balanced_accuracy, final_train_macro_f1, _, _ = run_epoch(
        model, train_loader, criterion, optimizer, device, train_mode=False,
    )
    test_loss, test_balanced_accuracy, test_macro_f1, y_true_test, y_pred_test = run_epoch(
        model, test_loader, criterion, optimizer, device, train_mode=False,
    )

    confusion = confusion_matrix(y_true_test, y_pred_test, labels=list(range(len(class_order))))
    pd.DataFrame(confusion, index=class_order, columns=class_order).to_csv(
        os.path.join(run_dir, f"confusion_matrix_fold{fold}.csv")
    )

    return {
        "fold": fold,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_sequences),
        "train_balanced_accuracy": final_train_balanced_accuracy,
        "val_balanced_accuracy": best_val_balanced_accuracy,
        "test_balanced_accuracy": test_balanced_accuracy,
        "test_macro_f1": test_macro_f1,
        "train_test_gap": final_train_balanced_accuracy - test_balanced_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="SAMuSe LSTM/GRU skill classification from joint positions.")
    parser.add_argument("--input_dir", required=True, help="Directory containing cleaned trial CSV files")
    parser.add_argument("--metadata_file", required=True, help="Participant metadata CSV/XLSX")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--instrument", choices=["violin", "clarinet", "all"], default="all")
    parser.add_argument("--task", choices=["2class", "3class"], required=True)
    parser.add_argument("--conditions", default="play,mime")
    parser.add_argument("--protocol", choices=["trial", "participant"], default="participant")
    parser.add_argument("--rnn_type", choices=["lstm", "gru"], default="gru")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    print(f"Using device: {device}")

    metadata_map = load_metadata(args.metadata_file)
    conditions = None if args.conditions.strip().lower() == "all" else {
        value.strip().lower() for value in args.conditions.split(",") if value.strip()
    }
    instrument_filter = None if args.instrument == "all" else args.instrument

    trial_table = collect_trials(
        args.input_dir, metadata_map, instrument_filter, args.task, conditions,
    )
    class_order = ["novice", "advanced_beginner", "expert"] if args.task == "3class" else ["novice", "expert"]

    print(f"Loaded {len(trial_table)} trials across {trial_table['participant_id'].nunique()} participants.")
    print("Class counts:", trial_table["label"].value_counts().to_dict())

    sequences, labels, participants, valid_rows = [], [], [], []
    for _, row in trial_table.iterrows():
        sequence = load_trial_sequence(row["file"], POSITION_COLUMNS)
        if sequence is None:
            continue
        sequences.append(sequence)
        labels.append(row["label"])
        participants.append(row["participant_id"])
        valid_rows.append(row)

    labels = np.array(labels)
    participants = np.array(participants)
    print(f"Usable trials after loading sequences: {len(sequences)}")

    if args.protocol == "trial":
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iterable = splitter.split(np.zeros(len(labels)), labels)
    else:
        splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iterable = splitter.split(np.zeros(len(labels)), labels, groups=participants)

    fold_metrics = []
    for fold, (train_index, test_index) in enumerate(split_iterable, start=1):
        train_sequences = [sequences[i] for i in train_index]
        train_labels = labels[train_index]
        test_sequences = [sequences[i] for i in test_index]
        test_labels = labels[test_index]
        train_participants_fold = participants[train_index]

        if args.protocol == "participant":
            train_participants_set = set(participants[train_index])
            test_participants_set = set(participants[test_index])
            overlap = train_participants_set & test_participants_set
            if overlap:
                raise RuntimeError(f"Fold {fold}: participant leakage detected: {sorted(overlap)}")

        metrics = train_one_fold(
            train_sequences, train_labels,
            test_sequences, test_labels,
            class_order, args, device, fold, args.output_dir,
            train_participants=train_participants_fold,
        )
        fold_metrics.append(metrics)
        print(f"Fold {fold} summary: {metrics}")

    fold_results = pd.DataFrame(fold_metrics)
    fold_results.to_csv(os.path.join(args.output_dir, "fold_metrics.csv"), index=False)

    summary = {
        "protocol": args.protocol,
        "instrument": args.instrument,
        "task": args.task,
        "rnn_type": args.rnn_type,
        "n_splits": args.n_splits,
        "test_balanced_accuracy_mean": float(fold_results["test_balanced_accuracy"].mean()),
        "test_balanced_accuracy_sd": float(fold_results["test_balanced_accuracy"].std(ddof=1)),
        "train_test_gap_mean": float(fold_results["train_test_gap"].mean()),
    }
    with open(os.path.join(args.output_dir, "experiment_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 88)
    print("EXPERIMENT SUMMARY — RNN skill classification")
    print("=" * 88)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved all outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

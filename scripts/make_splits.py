"""Create stratified train/val/test splits (70 train/10 val/20 test) for artefact classification.

Generates a single CSV file in <out_dir> depending on --version:
- --version 24 -> split_24.csv: original dataset restricted to the 24 classes in class_mapping.csv
  (without artefact exclusions and without taxonomy grouping).
- --version 7  -> split.csv: cleaned dataset applying:
    1. Exclusion of artefacts listed in .txt files (--exclude-txt).
    2. Removal of single images predicted as 'Other' from .csv files (--corrections-csv).
    3. Relabeling of artefacts with unanimous non-'Other' predictions in .csv files, or dropping
       artefacts with conflicting predictions across their images.
    4. Stratified split (70/10/20) on type_label, followed by taxonomy grouping according to
       new_label in class_mapping.csv.

Classes with fewer than 3 artefacts are automatically assigned entirely to the
'train' split so stratified splitting (70/10/20) succeeds on the remaining classes.

Output contains the columns: images_path,label,split,artefact

usage:
    python scripts/make_splits.py data/artefacts.csv data/class_mapping.csv --version 24
    python scripts/make_splits.py data/artefacts.csv data/class_mapping.csv --version 7 \
        --exclude-txt data/cleaning/excluded_artefacts_md5.txt \
        --corrections-csv data/cleaning/qwen_images_to_check.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_excluded_ids(txt_paths: list[Path]) -> set[str]:
    """Read Wikidata Q-codes from one or more newline-separated .txt files."""
    excluded = set()
    for path in txt_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Exclusion txt file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                code = line.strip()
                if code:
                    excluded.add(code.split("/")[-1])
    return excluded


def load_corrections_csv(csv_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate one or more prediction/cleaning CSV files."""
    if not csv_paths:
        return pd.DataFrame(columns=["image", "artefact", "label", "predicted"])

    frames = []
    required_cols = {"image", "artefact", "label", "predicted"}
    for path in csv_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Corrections CSV file not found: {path}")
        df_csv = pd.read_csv(path, keep_default_na=False)
        missing = required_cols - set(df_csv.columns)
        if missing:
            raise ValueError(f"Missing required columns {missing} in {path}")
        df_csv = df_csv[list(required_cols)].copy()
        df_csv["artefact"] = df_csv["artefact"].str.strip().str.split("/").str[-1]
        df_csv["image"] = df_csv["image"].str.strip()
        df_csv["predicted"] = df_csv["predicted"].str.strip()
        frames.append(df_csv)

    return pd.concat(frames, ignore_index=True)


def remove_other_images(df: pd.DataFrame, corrections_df: pd.DataFrame) -> pd.DataFrame:
    """Remove images predicted as 'Other' from the ';'-separated images_path of matching artefacts."""
    if corrections_df.empty:
        return df.copy()

    other_rows = corrections_df[corrections_df["predicted"].str.lower() == "other"]
    if other_rows.empty:
        return df.copy()

    other_rows["image"] = other_rows["image"].str.replace("\\", "/", regex=False)
    images_to_drop_by_artefact = (
        other_rows.groupby("artefact")["image"].apply(set).to_dict()
    )

    out = df.copy()
    mask = out["artefact"].isin(images_to_drop_by_artefact)
    if not mask.any():
        return out

    def _filter_paths(row: pd.Series) -> str:
        bad_images = images_to_drop_by_artefact[row["artefact"]]
        kept = [
            p.strip()
            for p in row["images_path"].split(";")
            if p.strip() and p.strip() not in bad_images
        ]
        return ";".join(kept)

    out.loc[mask, "images_path"] = out.loc[mask].apply(_filter_paths, axis=1)

    removed_images_count = len(other_rows)
    empty_after = int((out["images_path"].str.strip() == "").sum())
    print(
        f"[Info] Processed {removed_images_count} 'Other' image records across "
        f"{len(images_to_drop_by_artefact)} artefacts ({empty_after} artefacts left with 0 images)"
    )
    return out


def relabel_from_predictions(
    df: pd.DataFrame,
    corrections_df: pd.DataFrame,
    label_col: str,
    valid_labels: set[str],
) -> pd.DataFrame:
    """Update artefact labels when all non-'Other' predictions agree; drop conflicting artefacts."""
    if corrections_df.empty:
        return df.copy()

    non_other = corrections_df[corrections_df["predicted"].str.lower() != "other"].copy()
    if non_other.empty:
        return df.copy()

    # Check agreement per artefact across all non-'Other' image records
    pred_per_artefact = non_other.groupby("artefact")["predicted"].unique()

    unanimous_map: dict[str, str] = {}
    conflicting_artefacts: set[str] = set()

    for artefact_id, preds in pred_per_artefact.items():
        if len(preds) == 1:
            pred_label = preds[0]
            if pred_label in valid_labels:
                unanimous_map[artefact_id] = pred_label
            else:
                conflicting_artefacts.add(artefact_id)
        else:
            conflicting_artefacts.add(artefact_id)

    out = df.copy()

    # Relabel unanimous artefacts directly on type_label
    relabel_mask = out["artefact"].isin(unanimous_map)
    out.loc[relabel_mask, label_col] = out.loc[relabel_mask, "artefact"].map(unanimous_map)

    # Drop artefacts with conflicting predictions
    before_drop = len(out)
    out = out[~out["artefact"].isin(conflicting_artefacts)].copy()
    dropped_count = before_drop - len(out)

    print(
        f"[Info] CSV relabeling: updated {int(relabel_mask.sum())} artefacts with unanimous predictions, "
        f"dropped {dropped_count} artefacts with conflicting/invalid predictions"
    )
    return out


def stratified_split_artefacts(df: pd.DataFrame, stratify_col: str, seed: int = 42) -> pd.DataFrame:
    """Split artefact-level DataFrame into 70% train, 10% val, 20% test.

    - Splitting at artefact level before exploding images_path prevents data leakage
      across images belonging to the same artwork.
    - Classes with < 3 artefacts cannot be stratified across 3 splits and are placed
      entirely into 'train'.
    """
    df = df.copy()

    class_counts = df[stratify_col].value_counts()
    rare_classes = class_counts[class_counts < 3].index.tolist()

    rare_df = df[df[stratify_col].isin(rare_classes)].assign(split="train")
    common_df = df[~df[stratify_col].isin(rare_classes)].copy()

    if rare_classes:
        rare_summary = {cls: int(class_counts[cls]) for cls in rare_classes}
        print(
            f"[Info] Classes with < 3 artefacts assigned entirely to 'train' "
            f"(column '{stratify_col}'): {rare_summary}"
        )

    # 1. Split 80% (train+val) vs 20% test
    train_val_df, test_df = train_test_split(
        common_df,
        test_size=0.20,
        random_state=seed,
        stratify=common_df[stratify_col],
    )

    # 2. Split the remaining 80% into 70% train (7/8 of 80%) and 10% val (1/8 of 80%)
    val_relative_size = 0.10 / 0.80  # 0.125
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=seed,
        stratify=train_val_df[stratify_col],
    )

    train_df = pd.concat([train_df.assign(split="train"), rare_df], ignore_index=True)
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def expand_and_format(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Explode ';'-separated images_path and format final output columns."""
    out = df.copy()
    out["images_path"] = out["images_path"].str.split(";")
    out = out.explode("images_path").reset_index(drop=True)

    # Clean whitespace and remove any empty image paths
    out["images_path"] = out["images_path"].str.strip()
    out = out[out["images_path"] != ""].drop_duplicates(subset=["artefact", "images_path"]).copy()

    out["label"] = out[label_col]
    return out[["images_path", "label", "split", "artefact"]]


def main():
    parser = argparse.ArgumentParser(
        description="Generate either split_24.csv (--version 24) or split.csv (--version 7) from artefacts.csv"
    )
    parser.add_argument(
        "artefacts",
        type=Path,
        help="Path to raw artefacts.csv",
    )
    parser.add_argument(
        "mapping",
        type=Path,
        help="Path to class_mapping.csv (columns: old_label, new_label)",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Output directory for the generated CSV (default: data)",
    )
    parser.add_argument(
        "--version",
        choices=["24", "7"],
        required=True,
        help="Which split version to generate: '24' (split_24.csv) or '7' (split.csv)",
    )
    parser.add_argument(
        "--exclude-txt",
        type=Path,
        nargs="*",
        default=[Path("data/cleaning/excluded_artefacts_md5.txt")],
        help="One or more .txt files with artefact Q-codes to exclude first in --version 7",
    )
    parser.add_argument(
        "--corrections-csv",
        type=Path,
        nargs="*",
        default=[],
        help="One or more .csv files (image,artefact,label,predicted) to remove 'Other' images and relabel/drop artefacts in --version 7",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    args = parser.parse_args()

    # Load raw dataset and taxonomy mapping
    df = pd.read_csv(
        args.artefacts,
        usecols=["artefact", "type_label", "images_path"],
        keep_default_na=False,
    )
    mapping = pd.read_csv(args.mapping, keep_default_na=False)
    mapping["old_label"] = mapping["old_label"].str.strip()
    mapping["new_label"] = mapping["new_label"].str.strip()
    to_new_label = dict(zip(mapping["old_label"], mapping["new_label"]))
    valid_labels = set(mapping["old_label"].unique()) | set(mapping["new_label"].unique()) # union between old and new labels (check)

    df = df[df["images_path"].str.strip() != ""].copy()

    # Normalize artefact ID (e.g. http://www.wikidata.org/entity/Q1402761 -> Q1402761)
    # and keep only rows belonging to the 24 mapped classes
    df["artefact"] = df["artefact"].str.strip().str.split("/").str[-1]
    assert df["artefact"].is_unique, "Duplicate artefact IDs found in artefacts.csv"

    df["type_label"] = df["type_label"].str.strip()
    df = df[df["type_label"].isin(to_new_label)].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.version == "24":
        # ---------------------------------------------------------------------
        # Version 1: Original 24 classes, no exclusions, no taxonomy grouping
        # ---------------------------------------------------------------------
        print("=== Building Version 24: split_24.csv ===")
        df_v1 = stratified_split_artefacts(df, stratify_col="type_label", seed=args.seed)
        split_24 = expand_and_format(df_v1, label_col="type_label")
        out_24_path = args.out_dir / "split_24.csv"
        split_24.to_csv(out_24_path, index=False)

        print(f"Saved {out_24_path}: {len(split_24)} images across {split_24['label'].nunique()} classes")
        print(split_24["split"].value_counts().to_string())

    elif args.version == "7":
        # ---------------------------------------------------------------------
        # Version 2: Cleaned artefacts + CSV image removal/relabeling -> split -> 7 classes
        # ---------------------------------------------------------------------
        print("=== Building Version 7: split.csv ===")

        # 1. Remove artefacts listed in .txt exclusion files first
        excluded_ids = load_excluded_ids(args.exclude_txt) if args.exclude_txt else set()
        df_clean = df[~df["artefact"].isin(excluded_ids)].copy()
        print(
            f"[Info] Excluded {len(df) - len(df_clean)} artefacts using "
            f"{len(args.exclude_txt)} .txt file(s)"
        )

        # 2. Remove 'Other' images and relabel/drop artefacts directly on 'type_label'
        corrections_df = load_corrections_csv(args.corrections_csv)
        df_clean = remove_other_images(df_clean, corrections_df)
        df_clean = relabel_from_predictions(
            df_clean,
            corrections_df,
            label_col="type_label",
            valid_labels=valid_labels,
        )

        # Drop artefacts that lost all their images after removing 'Other'
        df_clean = df_clean[df_clean["images_path"].str.strip() != ""].copy()

        # 3. Stratified split on 'type_label' first, then map taxonomy to 7 classes
        df_v2 = stratified_split_artefacts(
            df_clean, stratify_col="type_label", seed=args.seed
        )
        df_v2["grouped_label"] = df_v2["type_label"].replace(to_new_label)

        # 4. Explode images_path and format output
        split_grouped = expand_and_format(df_v2, label_col="grouped_label")
        out_grouped_path = args.out_dir / "split.csv"
        split_grouped.to_csv(out_grouped_path, index=False)

        print(f"Saved {out_grouped_path}: {len(split_grouped)} images across {split_grouped['label'].nunique()} classes")
        print(split_grouped["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
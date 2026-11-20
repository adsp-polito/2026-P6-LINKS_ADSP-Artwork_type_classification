"""Find artefacts to drop because they share images (same md5) with other artefacts.

- same image in artefacts with the same class: keep the artefact with more images
- same image in artefacts with different classes: drop all of them

Writes the ids of the artefacts to drop, one per line, to <out_dir>/excluded_artefacts_md5.txt.

usage: python scripts/md5_duplicates.py <dataset>/artefacts.csv data/class_mapping.csv <dataset> data/cleaning
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def md5(path):
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artefacts")
    parser.add_argument("mapping")
    parser.add_argument("images_root")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    df = pd.read_csv(args.artefacts, usecols=["artefact", "type_label", "images_path"], keep_default_na=False)
    mapping = pd.read_csv(args.mapping, keep_default_na=False)
    to_class = dict(zip(mapping["old_label"].str.strip(), mapping["new_label"].str.strip()))
    df["label"] = df["type_label"].map(to_class)
    df = df.dropna(subset=["label"]).copy()
    df["artefact"] = df["artefact"].str.split("/").str[-1]
    df["images_path"] = df["images_path"].str.split(";")

    images = df.explode("images_path").reset_index(drop=True)
    images["n_images"] = images.groupby("artefact")["images_path"].transform("size")

    root = Path(args.images_root)
    images["md5"] = [md5(root / p) for p in tqdm(images["images_path"])]

    # skip missing files, and count an image repeated inside one artefact only once
    images = images.dropna(subset=["md5"]).drop_duplicates(["md5", "artefact"])

    shared = images[images.duplicated("md5", keep=False)]
    shared = shared.sort_values(["md5", "n_images", "artefact"], ascending=[True, False, True]).copy()
    shared["conflict"] = shared.groupby("md5")["label"].transform("nunique") > 1
    drop = shared["conflict"] | shared.duplicated("md5")

    # an artefact can be in several groups: drop it if any group drops it
    dropped = shared[drop].sort_values("conflict", ascending=False).drop_duplicates("artefact")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "excluded_artefacts_md5.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(dropped["artefact"])) + "\n")

    print(f"dropped {len(dropped)} artefacts ({dropped['n_images'].sum()} images), "
          f"{int(dropped['conflict'].sum())} because of a label conflict")
    per_class = pd.DataFrame({
        "artefacts": df["label"].value_counts(),
        "dropped": dropped["label"].value_counts(),
    }).fillna(0).astype(int).sort_values("dropped", ascending=False)
    print("\ndropped artefacts per class:")
    print(per_class.to_string())


if __name__ == "__main__":
    main()

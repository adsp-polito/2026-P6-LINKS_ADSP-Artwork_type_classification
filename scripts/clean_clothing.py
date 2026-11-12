"""Remove depictions of clothing (sketches, prints, photographs, postcards) from the clothing branch.

Many artefacts typed as Clothing (or one of its subclasses) are not garments but
representations of garments: fashion sketches on paper, 18th-century fashion
plates, photographs of people in costume, and postcards mislabelled as Clothing.
This script splits artefacts.csv into:

    artefacts_clean.csv    artefacts kept (same columns as the input)
    artefacts_removed.csv  artefacts removed (same columns as the input)

Only the clothing branch is filtered; all other types are kept unchanged.
The rules use metadata only (materials, label, description), not the images.

Usage:
    python scripts/clean_clothing.py path/to/artefacts.csv path/to/out_dir [--dry-run]

With --dry-run the script only prints the counts and a sample of removed
photographs and postcards, without writing any file.
"""
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

# Clothing branch of the CACAO taxonomy, as it appears in the type_label column.
CLOTHING = {
    "Clothing", "Main Garments", "Dresses", "Coats", "Robes", "Suits", "Tunics",
    "Outerwear", "Cloaks", "Underwear", "Chemises", "Underpants", "Underdresses",
    "Lingerie", "Underskirts",
}

# Keywords (lowercase) searched in label + description, in the languages found in the data.
PHOTO_WORDS = ("photograph", "fotograf", "clișeu", "clişeu", "cliseu", "negativ")
POSTCARD_WORDS = ("hilmteich", "carte poștală", "carte postala", "ansichtskarte",
                  "postcard", "carte postale", "vedere de", "vedere din", "vedere asupra")
SKETCH_WORDS = ("croquis", "dessin", "crayon", "encre", "inscription", "manuscrit", "modèle",
                "drawing", "sketch", "album")
PRINT_WORDS = ("recto", "verso", "montage", "carnavalet", "cah.", "estampe")

# Keywords (lowercase) searched in materials_label.
TEXTILE_WORDS = ("cotton", "wool", "silk", "linen", "fabric", "cloth", "thread", "embroider",
                 "weav", "lace", "sewing", "hemp", "leather", "fur", "velvet", "crochet", "knit")


def removal_reason(row):
    """Return why the artefact is removed ('postcard', 'photograph', 'sketch', 'print'), or None to keep it.

    Rules are applied in order; the first one that matches decides.
    """
    if row["type_label"] not in CLOTHING:
        return None
    # The "ST..." collection contains real garments (some list paper among their materials).
    if re.match(r"^ST\d", row["label"]):
        return None

    materials = row["materials_label"].lower()
    text = (row["label"] + " " + row["description"]).lower()
    is_textile = any(w in materials for w in TEXTILE_WORDS)

    if any(w in text for w in POSTCARD_WORDS):
        return "postcard"
    if any(w in materials for w in ("photograph", "photo-sensitized")):
        return "photograph"
    # Text-only match: skip textile objects, e.g. a real shirt whose description
    # mentions it was photographed for the inventory ("la fotografiere").
    if any(w in text for w in PHOTO_WORDS) and not is_textile:
        return "photograph"
    if "paper" in materials or "cardboard" in materials:
        # Real garment with paper parts: textile materials, no paint, no drawing vocabulary.
        if is_textile and "paint" not in materials and not any(w in text for w in SKETCH_WORDS):
            return None
        return "sketch"
    # Fashion plates (e.g. Musée Carnavalet) have no materials but describe a mounted print.
    if not materials.strip() and any(w in text for w in PRINT_WORDS):
        return "print"
    return None


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    dry_run = "--dry-run" in sys.argv
    sys.stdout.reconfigure(encoding="utf-8")
    csv.field_size_limit(10**9)

    # Keep the line endings of the input file.
    with open(src, "rb") as f:
        eol = "\r\n" if b"\r\n" in f.read(100_000) else "\n"

    with open(src, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    kept, removed, reasons = [], [], Counter()
    for row in rows:
        reason = removal_reason(row)
        if reason:
            removed.append(row)
            reasons[reason] += 1
        else:
            kept.append(row)

    print(f"Total {len(rows)} | kept {len(kept)} | removed {len(removed)}")
    print("Removed by reason:", dict(reasons))
    print("Removed by type:", dict(Counter(r["type_label"] for r in removed).most_common()))
    print("Clothing kept by type:",
          dict(Counter(r["type_label"] for r in kept if r["type_label"] in CLOTHING).most_common()))

    if dry_run:
        rng = random.Random(0)
        for reason in ("photograph", "postcard"):
            sample = [r for r in removed if removal_reason(r) == reason]
            print(f"\n-- sample: {reason} ({len(sample)})")
            for r in rng.sample(sample, min(25, len(sample))):
                print(" *", r["type_label"], "|", r["label"][:40], "|",
                      r["description"][:110].replace("\n", " "), "|", r["materials_label"][:40])
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("artefacts_clean.csv", kept), ("artefacts_removed.csv", removed)):
        with open(out_dir / name, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator=eol)
            writer.writeheader()
            writer.writerows(data)
        print(f"Wrote {out_dir / name} ({len(data)} rows)")


if __name__ == "__main__":
    main()

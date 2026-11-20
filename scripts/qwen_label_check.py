import csv
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

DATASET = Path("/kaggle/input/datasets/fabiostradiotti/artkb-dataset")
ARTEFACTS = DATASET / "artefacts.csv"
IMAGES_ROOT = DATASET
MAPPING = Path("data/class_mapping.csv")
EXCLUDE = Path("data/cleaning/excluded_artefacts_md5.txt")
OUT_DIR = Path("/kaggle/working")
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BATCH_SIZE = 32
MAX_SIDE = 768  # downscale images to limit memory and time

CLASS_DESCRIPTIONS = {
    "Painting": "a painted artwork (on canvas, panel, paper, or a wall fresco)",
    "Buildings": "architecture: the exterior or interior of a building or site, only photo of the real building, not a painting or model of it."
                 " (church, mosque, temple, tomb, villa, ruins, necropolis)",
    "Instruments": "an instrument, e.g. a musical or scientific instrument, a church organ, if the instrument is the main subject of the image",
    "Clothing": "a garment or item of clothing (dress, coat, cloak, tunic, suit, underwear), worn or displayed",
    "Medals": "a medal or medallion (either side)",
    "Mosaics": "a picture or pattern made of small coloured tiles or stones (tesserae)",
    "Documents": "a written or printed document, such as a letter or a manuscript",
}
PROMPT = (
    "You are an expert in cultural heritage cataloguing. "
    "Classify the type of artefact shown in this image of a museum artefact.\n"
    "Choose exactly one of these classes:\n"
    + "\n".join(f"- {c}: {d}" for c, d in CLASS_DESCRIPTIONS.items())
    + "\n- Other: none of the above.\n\n"
    "Answer with the class name only, nothing else."
)


def load_images():
    """One row per image: image, artefact, label."""
    df = pd.read_csv(ARTEFACTS, usecols=["artefact", "type_label", "images_path"], keep_default_na=False)
    mapping = pd.read_csv(MAPPING, keep_default_na=False)
    to_class = dict(zip(mapping["old_label"].str.strip(), mapping["new_label"].str.strip()))
    df["label"] = df["type_label"].str.strip().map(to_class)
    df["artefact"] = df["artefact"].str.split("/").str[-1]
    excluded = set(EXCLUDE.read_text(encoding="utf-8").split())
    df = df[df["label"].notna() & ~df["artefact"].isin(excluded)]

    df = df.assign(image=df["images_path"].str.split(";")).explode("image")
    df["image"] = df["image"].str.strip()
    return df[df["image"] != ""][["image", "artefact", "label"]]


def open_image(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
    except OSError:
        return None
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def load_model():
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left" #to generate in batch
    return model.eval(), processor


def predict(model, processor, images):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text] * len(images), images=images, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    answers = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    classes = {c.lower(): c for c in [*CLASS_DESCRIPTIONS, "Other"]}
    return [classes.get(a.strip().strip(".").lower(), a.strip()) for a in answers]


def run(todo, pred_file):
    model, processor = load_model()
    n_to_check = 0
    with open(pred_file, "a", newline="", encoding="utf-8") as f, \
            tqdm(total=len(todo), desc="classifying", unit="img") as bar:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["image", "artefact", "label", "predicted"])
        for start in range(0, len(todo), BATCH_SIZE):
            rows = list(todo.iloc[start:start + BATCH_SIZE].itertuples())
            images = [open_image(IMAGES_ROOT / r.image) for r in rows]
            ok = [i for i, img in enumerate(images) if img is not None]
            answers = dict(zip(ok, predict(model, processor, [images[i] for i in ok]))) if ok else {}
            for i, r in enumerate(rows):
                predicted = answers.get(i, "UNREADABLE")
                writer.writerow([r.image, r.artefact, r.label, predicted])
                n_to_check += predicted != r.label
            f.flush()
            bar.update(len(rows))
            bar.set_postfix(to_check=n_to_check)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    classes = sorted(c.strip() for c in os.environ.get("CLASSES", "").split(",") if c.strip())
    suffix = "_".join(classes) if classes else "all"  # output files are named after the classes of the run
    pred_file = OUT_DIR / f"qwen_predictions_{suffix}.csv"
    check_file = OUT_DIR / f"qwen_images_to_check_{suffix}.csv"

    images = load_images()
    done = set(pd.read_csv(pred_file)["image"]) if pred_file.exists() else set()
    todo = images[~images["image"].isin(done)]
    if classes:
        todo = todo[todo["label"].isin(classes)]
    per_class = int(os.environ.get("PER_CLASS", 0))
    if per_class:
        todo = todo.sample(frac=1, random_state=0).groupby("label").head(per_class)
    print(f"{len(images)} images, {len(todo)} still to classify")
    if len(todo):
        run(todo, pred_file)

    res = pd.read_csv(pred_file)
    print(pd.crosstab(res["label"], res["predicted"]).to_string())
    to_check = res[res["predicted"] != res["label"]].sort_values(["label", "predicted", "image"])
    to_check.to_csv(check_file, index=False)
    print(f"{len(to_check)} of {len(res)} images to check, saved to {check_file}")


if __name__ == "__main__":
    main()

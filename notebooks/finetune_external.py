"""Fine-tune FreshSight MobileNetV3 dengan data eksternal.

Aturan eksperimen:
- Test set asli (997 objek) TIDAK disentuh.
- AFQC: Train/apple_fresh + Train/apple_rotten masuk training secara seimbang.
  Ini diperlukan agar model tidak collapse ke rotten_apple pada domain AFQC.
- Banana_images_CNN: hanya folder "over ripe" -> rotten_banana.
- Split banana dilakukan per klaster dHash agar tidak bocor ke external test.
- Class weight dihitung ulang dari combined train.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
DHASH_THR = 2
SIZE_TOL = 0.10

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "materials" / "dataset"
AFQC_DIR = DATASET_DIR / "AFQC_apple"
BANANA_DIR = DATASET_DIR / "Banana_overripe" / "over ripe"
CROPS_EXT_DIR = ROOT / "data" / "crops_external"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
for d in (CROPS_EXT_DIR, REPORT_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ["fresh_apple", "rotten_apple", "fresh_banana", "rotten_banana", "fresh_orange", "rotten_orange"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

np.random.seed(SEED)
tf.random.set_seed(SEED)


def dhash(im, size=8):
    small = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def image_files(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def prepare_external_image(src, dst):
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        canvas.paste(im, ((IMG_SIZE - im.width) // 2, (IMG_SIZE - im.height) // 2))
        dst.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dst, quality=92)


def build_original_hashes():
    rows = []
    img_dir = DATASET_DIR / "JPEGImages"
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        try:
            with Image.open(p) as im:
                rows.append({"path": str(p), "dhash": dhash(im), "w": im.width, "h": im.height})
        except Exception:
            pass
    return pd.DataFrame(rows)


def is_near_dup(h, w, hh, hash_arr, widths, heights):
    d = popcount64(hash_arr ^ np.uint64(h))
    for j in np.where(d <= DHASH_THR)[0]:
        if (abs(w - widths[j]) / max(w, widths[j]) <= SIZE_TOL and
                abs(hh - heights[j]) / max(hh, heights[j]) <= SIZE_TOL):
            return True
    return False


def banana_cluster_split(files):
    hashes, widths, heights = [], [], []
    for p in files:
        with Image.open(p) as im:
            hashes.append(dhash(im)); widths.append(im.width); heights.append(im.height)
    hashes = np.array(hashes, dtype=np.uint64)
    widths = np.array(widths, dtype=float); heights = np.array(heights, dtype=float)
    parent = list(range(len(files)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(files) - 1):
        d = popcount64(hashes[i + 1:] ^ hashes[i])
        for off in np.where(d <= DHASH_THR)[0]:
            j = i + 1 + int(off)
            if (abs(widths[i] - widths[j]) / max(widths[i], widths[j]) <= SIZE_TOL and
                    abs(heights[i] - heights[j]) / max(heights[i], heights[j]) <= SIZE_TOL):
                union(i, j)
    clusters = {}
    for i, p in enumerate(files):
        clusters.setdefault(find(i), []).append(p)
    cids = list(clusters)
    rng = np.random.RandomState(SEED)
    rng.shuffle(cids)
    total = len(files)
    out = {}
    cum = 0
    for cid in cids:
        group = clusters[cid]
        frac = (cum + len(group) / 2) / total
        split = "train" if frac < 0.80 else ("val" if frac < 0.90 else "external_test")
        for p in group:
            out[p] = split
        cum += len(group)
    return out


def prepare_external(force=False):
    orig = build_original_hashes()
    oh = orig.dhash.to_numpy(dtype=np.uint64)
    ow = orig.w.to_numpy(dtype=float); ohgt = orig.h.to_numpy(dtype=float)
    print(f"Original hashes: {len(orig)}")

    records = []

    # AFQC: fresh + rotten untuk train (seimbang); Test fresh + rotten untuk external test.
    for split_name, subdir, label6 in [
        ("train", "Train/apple_fresh", "fresh_apple"),
        ("train", "Train/apple_rotten", "rotten_apple"),
        ("external_test", "Test/apple_fresh", "fresh_apple"),
        ("external_test", "Test/apple_rotten", "rotten_apple"),
    ]:
        folder = AFQC_DIR / subdir
        files = image_files(folder)
        print(subdir, len(files))
        for p in files:
            try:
                with Image.open(p) as im:
                    h, w, hh = dhash(im), im.width, im.height
            except Exception as e:
                print("skip corrupt", p, e); continue
            if is_near_dup(h, w, hh, oh, ow, ohgt):
                continue
            records.append({
                "filename": str(p.relative_to(ROOT)), "source": "AFQC", "source_split": subdir,
                "label6": label6, "split": split_name, "external": 1,
            })

    # Banana: hanya folder over ripe.
    banana_files = image_files(BANANA_DIR)
    print("Banana over ripe:", len(banana_files))
    banana_split = banana_cluster_split(banana_files)
    for p in banana_files:
        try:
            with Image.open(p) as im:
                h, w, hh = dhash(im), im.width, im.height
        except Exception as e:
            print("skip corrupt", p, e); continue
        if is_near_dup(h, w, hh, oh, ow, ohgt):
            continue
        records.append({
            "filename": str(p.relative_to(ROOT)), "source": "Banana_overripe",
            "source_split": str(p.parent.relative_to(BANANA_DIR.parent)),
            "label6": "rotten_banana", "split": banana_split[p], "external": 1,
        })

    ext = pd.DataFrame(records)
    ext["label_id"] = ext.label6.map(CLASS_TO_ID)
    # buang duplikat near-duplicate internal external (pertahankan test)
    ext = ext.sort_values("split", key=lambda s: s.map({"train": 0, "val": 1, "external_test": 2}))
    keep = []
    kept = []
    for _, row in ext.iterrows():
        with Image.open(ROOT / row.filename) as im:
            h, w, hh = dhash(im), im.width, im.height
        dup = False
        for k in kept:
            if popcount64(np.uint64(h) ^ np.uint64(k["hash"])) <= DHASH_THR:
                if (abs(w - k["w"]) / max(w, k["w"]) <= SIZE_TOL and
                        abs(hh - k["h"]) / max(hh, k["h"]) <= SIZE_TOL):
                    dup = True
                    break
        if not dup:
            row = row.copy()
            row["crop_path"] = None
            keep.append(row)
            kept.append({"hash": h, "w": w, "h": hh})
    ext = pd.DataFrame(keep).reset_index(drop=True)

    # tulis crop 224x224
    for i, row in ext.iterrows():
        src = ROOT / row["filename"]
        out = CROPS_EXT_DIR / row["split"] / row["label6"] / f"{row['source'].lower()}__{i:05d}.jpg"
        if force or not out.exists():
            prepare_external_image(src, out)
        ext.at[i, "crop_path"] = out.relative_to(ROOT).as_posix()

    ext.to_csv(REPORT_DIR / "08_external_manifest.csv", index=False)
    print("External manifest:", len(ext))
    print(ext.groupby(["split", "label6"]).size().to_string())
    return ext


def read_original_manifest():
    df = pd.read_csv(REPORT_DIR / "02_crops_manifest.csv")
    for col in ("label_id",):
        df[col] = df[col].astype(int)
    return df


def make_ds(df, training=False):
    paths = [str(ROOT / p) for p in df["crop_path"]]
    labels = df["label_id"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04, fill_mode="reflect"),
            tf.keras.layers.RandomZoom(0.08, fill_mode="reflect"),
        ], name="augmentasi_eksternal")

        def augment(image, label):
            image = aug(image, training=True)
            image = tf.image.random_brightness(image, max_delta=0.08)
            image = tf.image.random_contrast(image, lower=0.92, upper=1.08)
            return tf.clip_by_value(image, 0.0, 1.0), label

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def metrics_for(model, ds, df, title):
    probs = model.predict(ds, verbose=0)
    y_true = df["label_id"].to_numpy(dtype=int)
    y_pred = probs.argmax(axis=1)
    rep = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))),
                                target_names=CLASSES, output_dict=True, zero_division=0)
    out = {
        "name": title,
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {c: {k: float(v) for k, v in rep[c].items()} for c in CLASSES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist(),
    }
    pred = df.copy()
    pred["pred_label"] = [CLASSES[i] for i in y_pred]
    pred["confidence"] = probs.max(axis=1)
    return out, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    ext = prepare_external()
    if args.prepare_only:
        return

    original = read_original_manifest()
    orig_train = original[original.split == "train"].copy()
    orig_val = original[original.split == "val"].copy()
    orig_test = original[original.split == "test"].copy()

    ext_train = ext[ext.split == "train"].copy()
    ext_test = ext[ext.split == "external_test"].copy()

    train = pd.concat([orig_train, ext_train], ignore_index=True)
    val = orig_val.copy()
    test = orig_test.copy()

    counts = train["label6"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    weights = len(train) / (len(CLASSES) * counts.astype(float))
    cw = {CLASS_TO_ID[c]: float(w) for c, w in weights.items()}
    summary = {
        "original_train": int(len(orig_train)),
        "original_val": int(len(orig_val)),
        "original_test": int(len(orig_test)),
        "added_afqc_fresh_apple": int(((ext_train.source == "AFQC") & (ext_train.label6 == "fresh_apple")).sum()),
        "added_afqc_rotten_apple": int(((ext_train.source == "AFQC") & (ext_train.label6 == "rotten_apple")).sum()),
        "added_afqc_total": int((ext_train.source == "AFQC").sum()),
        "added_banana_overripe": int((ext_train.source == "Banana_overripe").sum()),
        "external_test_total": int(len(ext_test)),
        "external_test_by_source": {
            f"{src}|{lab}": int(n)
            for (src, lab), n in ext_test.groupby(["source", "label6"]).size().items()
        },
        "combined_train": int(len(train)),
        "train_counts": {c: int(counts[c]) for c in CLASSES},
        "class_weight": {c: round(float(weights[c]), 4) for c in CLASSES},
        "imbalance_ratio_before": round(float(orig_train.label6.value_counts().max() /
                                                 orig_train.label6.value_counts().min()), 3),
        "imbalance_ratio_after": round(float(counts.max() / max(counts.min(), 1)), 3),
    }
    (REPORT_DIR / "08_external_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"label": CLASSES, "train": [counts[c] for c in CLASSES],
                  "class_weight": [weights[c] for c in CLASSES]}).to_csv(
        REPORT_DIR / "08_external_class_distribution.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    train_ds = make_ds(train, training=True)
    val_ds = make_ds(val, training=False)
    test_ds = make_ds(test, training=False)
    ext_test_ds = make_ds(ext_test, training=False)

    model = tf.keras.models.load_model(MODEL_DIR / "mobilenetv3.keras", compile=False)
    baseline_orig, _ = metrics_for(model, test_ds, test, "baseline_original_test")
    baseline_ext, _ = metrics_for(model, ext_test_ds, ext_test, "baseline_external_test")

    backbone = model.get_layer("MobilenetV3small")
    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    for layer in model.layers:
        if layer.name in ("gap", "drop", "prediksi", "ke_minus1_1"):
            layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    ckpt = MODEL_DIR / "mobilenetv3_external.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     class_weight=cw, callbacks=callbacks, verbose=1)
    pd.DataFrame(hist.history).to_csv(REPORT_DIR / "08_external_history.csv", index=False)

    best = tf.keras.models.load_model(ckpt, compile=False)
    new_orig, pred_orig = metrics_for(best, test_ds, test, "finetuned_original_test")
    new_ext, pred_ext = metrics_for(best, ext_test_ds, ext_test, "finetuned_external_test")
    pred_orig.to_csv(REPORT_DIR / "08_original_test_predictions.csv", index=False)
    pred_ext.to_csv(REPORT_DIR / "08_external_test_predictions.csv", index=False)

    report = {
        "experiment": "original + AFQC apple fresh/rotten + banana overripe",
        "summary": summary,
        "baseline": {"original_test": baseline_orig, "external_test": baseline_ext},
        "finetuned": {"original_test": new_orig, "external_test": new_ext},
        "epochs_run": len(hist.history.get("loss", [])),
    }
    (REPORT_DIR / "08_external_experiment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([
        {"model": "baseline", "test": "original", "accuracy": baseline_orig["accuracy"],
         "macro_f1": baseline_orig["macro_f1"]},
        {"model": "finetuned", "test": "original", "accuracy": new_orig["accuracy"],
         "macro_f1": new_orig["macro_f1"]},
        {"model": "baseline", "test": "external", "accuracy": baseline_ext["accuracy"],
         "macro_f1": baseline_ext["macro_f1"]},
        {"model": "finetuned", "test": "external", "accuracy": new_ext["accuracy"],
         "macro_f1": new_ext["macro_f1"]},
    ]).to_csv(REPORT_DIR / "08_external_comparison.csv", index=False)

    print("\n=== RINGKASAN ===")
    print(pd.DataFrame([
        ["baseline", "original", baseline_orig["accuracy"], baseline_orig["macro_f1"]],
        ["finetuned", "original", new_orig["accuracy"], new_orig["macro_f1"]],
        ["baseline", "external", baseline_ext["accuracy"], baseline_ext["macro_f1"]],
        ["finetuned", "external", new_ext["accuracy"], new_ext["macro_f1"]],
    ], columns=["model", "test", "accuracy", "macro_f1"]).to_string(index=False))
    print("Model:", ckpt)
    print("Report:", REPORT_DIR / "08_external_experiment.json")


if __name__ == "__main__":
    main()

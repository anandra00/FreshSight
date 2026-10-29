import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(r"C:\Users\anand\OneDrive\Documents\Semester 5\Pemrosesan CItra\PJBL\Citra + Viskom")
IMG_SIZE = 224


def load(path):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        return np.asarray(im, dtype=np.float32) / 255.0


def evaluate(model_path, df):
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = []
    for p in df.crop_path:
        x = load(ROOT / p)[None].astype(np.float32)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        preds.append(int(interp.get_tensor(out["index"])[0].argmax()))
    y = df.label_id.to_numpy(dtype=int)
    return {"accuracy": float(accuracy_score(y, preds)),
            "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0))}


orig = pd.read_csv(ROOT / "reports" / "02_crops_manifest.csv")
test = orig[orig.split == "test"].copy()
ext = pd.read_csv(ROOT / "reports" / "08_external_manifest.csv")
ext_test = ext[ext.split == "external_test"].copy()

results = {}
for name in ["mobilenetv3_external_float32.tflite", "mobilenetv3_external_dynamic.tflite"]:
    p = ROOT / "models" / name
    results[name] = {"original_test": evaluate(p, test), "external_test": evaluate(p, ext_test)}
    print(name, results[name])
(ROOT / "reports" / "08_external_tflite_metrics.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")

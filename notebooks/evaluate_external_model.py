"""Evaluate baseline vs fine-tuned model on frozen original test and external test."""

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from finetune_external import (CLASSES, MODEL_DIR, REPORT_DIR, ROOT, metrics_for,
                               make_ds, read_original_manifest)

ext = pd.read_csv(REPORT_DIR / "08_external_manifest.csv")
orig = read_original_manifest()
test = orig[orig.split == "test"].copy()
ext_test = ext[ext.split == "external_test"].copy()

test_ds = make_ds(test, training=False)
ext_ds = make_ds(ext_test, training=False)

base = tf.keras.models.load_model(MODEL_DIR / "mobilenetv3.keras", compile=False)
fine = tf.keras.models.load_model(MODEL_DIR / "mobilenetv3_external.keras", compile=False)

b_orig, _ = metrics_for(base, test_ds, test, "baseline_original_test")
b_ext, _ = metrics_for(base, ext_ds, ext_test, "baseline_external_test")
f_orig, p_orig = metrics_for(fine, test_ds, test, "finetuned_original_test")
f_ext, p_ext = metrics_for(fine, ext_ds, ext_test, "finetuned_external_test")

p_orig.to_csv(REPORT_DIR / "08_original_test_predictions.csv", index=False)
p_ext.to_csv(REPORT_DIR / "08_external_test_predictions.csv", index=False)

summary = json.loads((REPORT_DIR / "08_external_dataset_summary.json").read_text(encoding="utf-8"))
report = {
    "experiment": "original + AFQC apple fresh/rotten + banana overripe",
    "summary": summary,
    "baseline": {"original_test": b_orig, "external_test": b_ext},
    "finetuned": {"original_test": f_orig, "external_test": f_ext},
}
(REPORT_DIR / "08_external_experiment.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
pd.DataFrame([
    {"model": "baseline", "test": "original", "accuracy": b_orig["accuracy"], "macro_f1": b_orig["macro_f1"]},
    {"model": "finetuned", "test": "original", "accuracy": f_orig["accuracy"], "macro_f1": f_orig["macro_f1"]},
    {"model": "baseline", "test": "external", "accuracy": b_ext["accuracy"], "macro_f1": b_ext["macro_f1"]},
    {"model": "finetuned", "test": "external", "accuracy": f_ext["accuracy"], "macro_f1": f_ext["macro_f1"]},
]).to_csv(REPORT_DIR / "08_external_comparison.csv", index=False)

print(pd.DataFrame([
    ["baseline", "original", b_orig["accuracy"], b_orig["macro_f1"]],
    ["finetuned", "original", f_orig["accuracy"], f_orig["macro_f1"]],
    ["baseline", "external", b_ext["accuracy"], b_ext["macro_f1"]],
    ["finetuned", "external", f_ext["accuracy"], f_ext["macro_f1"]],
], columns=["model", "test", "accuracy", "macro_f1"]).to_string(index=False))
print("\nExternal per class:")
for c in CLASSES:
    print(c, f_ext["per_class"][c])

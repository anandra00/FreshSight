"""Evaluate baseline vs external-data model on the 16 Wikimedia images."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

ROOT = Path(r"C:\Users\anand\OneDrive\Documents\Semester 5\Pemrosesan CItra\PJBL\Citra + Viskom")
IMG_SIZE = 224
CLASSES = ["fresh_apple", "rotten_apple", "fresh_banana", "rotten_banana", "fresh_orange", "rotten_orange"]


def auto_crop_buah(im, pad=0.08):
    rgb = np.asarray(im.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(0.04 * w), int(0.04 * h), int(0.92 * w), int(0.92 * h))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return im
    fg = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return im
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y = int(stats[idx, cv2.CC_STAT_LEFT]), int(stats[idx, cv2.CC_STAT_TOP])
    bw, bh = int(stats[idx, cv2.CC_STAT_WIDTH]), int(stats[idx, cv2.CC_STAT_HEIGHT])
    if bw * bh < 0.05 * w * h:
        return im
    p = int(pad * max(bw, bh))
    x1, y1 = max(0, x - p), max(0, y - p)
    x2, y2 = min(w, x + bw + p), min(h, y + bh + p)
    return im.crop((x1, y1, x2, y2))


def predict(model, im):
    arr = np.asarray(im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
                     dtype=np.float32) / 255.0
    prob = model.predict(arr[None], verbose=0)[0]
    return CLASSES[int(prob.argmax())], float(prob.max())


def main():
    base = tf.keras.models.load_model(ROOT / "models" / "mobilenetv3.keras", compile=False)
    fine = tf.keras.models.load_model(ROOT / "models" / "mobilenetv3_external.keras", compile=False)
    ref = pd.read_csv(ROOT / "reports" / "07_external_test.csv")
    rows = []
    for _, r in ref.iterrows():
        p = ROOT / "data" / "external_test" / r["file"]
        if not p.exists():
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            im_auto = auto_crop_buah(im)
        bp, bc = predict(base, im)
        ba, bac = predict(base, im_auto)
        fp, fc = predict(fine, im)
        fa, fac = predict(fine, im_auto)
        rows.append({
            "file": r["file"], "true": r["true"],
            "baseline": bp, "baseline_conf": bc, "baseline_ok": bp == r["true"],
            "baseline_auto": ba, "baseline_auto_conf": bac, "baseline_auto_ok": ba == r["true"],
            "finetuned": fp, "finetuned_conf": fc, "finetuned_ok": fp == r["true"],
            "finetuned_auto": fa, "finetuned_auto_conf": fac, "finetuned_auto_ok": fa == r["true"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "reports" / "09_wikimedia_external.csv", index=False)
    cols = ["baseline_ok", "baseline_auto_ok", "finetuned_ok", "finetuned_auto_ok"]
    print(out[["file", "true", "baseline", "baseline_auto", "finetuned", "finetuned_auto"]].to_string(index=False))
    print("\nAccuracy:")
    print(out[cols].mean().round(4).to_string())
    print("\nCorrect counts:")
    print(out[cols].sum().to_string())


if __name__ == "__main__":
    main()

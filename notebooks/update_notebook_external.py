import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\anand\OneDrive\Documents\Semester 5\Pemrosesan CItra\PJBL\Citra + Viskom")
nb_path = ROOT / "notebooks" / "FreshSight.ipynb"
backup = Path(r"C:\Users\anand\AppData\Local\Temp\FreshSight_before_external.ipynb")
shutil.copy2(nb_path, backup)

nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Update status table in cell 0.
c0 = nb["cells"][0]
src0 = "".join(c0.get("source", []))
old_row = "| 7 | Inferensi `predict_and_explain()` | selesai |"
new_rows = old_row + "\n| 8 | Uji generalisasi eksternal & auto-crop | selesai |\n| 9 | Eksperimen data eksternal (AFQC + banana overripe) | selesai |"
if old_row in src0 and "| 9 |" not in src0:
    src0 = src0.replace(old_row, new_rows)
    c0["source"] = src0

md1 = """## 9. Eksperimen data eksternal (AFQC + banana overripe)

Eksperimen ini menambahkan data eksternal **hanya ke training/validation**. Test set asli 997 objek tetap beku.

- **AFQC apple:** 1.011 `fresh_apple` + 1.028 `rotten_apple` setelah dedupe.
- **Banana overripe:** hanya folder `over ripe`; 954 train, 119 val, 119 external test.
- **External test:** 344 gambar (116 fresh apple, 109 rotten apple, 119 rotten banana).
- **Imbalance:** 12,67× → **9,77×**; class weight dihitung ulang dari combined train.
- **Model:** MobileNetV3 fine-tune dari `models/mobilenetv3.keras`, 6 epoch maksimum, EarlyStopping.

| Model | Original test (997) | External test (344) |
|---|---:|---:|
| MobileNetV3 baseline | 99,20% / macro F1 0,9871 | 62,21% / macro F1 0,3223 |
| MobileNetV3 + data eksternal | **99,60% / macro F1 0,9912** | **90,99% / macro F1 0,9071** |

Per kelas external test: `fresh_apple` F1 0,846 (recall 73,3%), `rotten_apple` F1 0,876 (recall 100%), `rotten_banana` F1 1,000. External test ini tidak memuat fresh banana/orange, jadi macro F1 tidak sebanding langsung dengan test asli.

Pada 16 gambar Wikimedia yang lama: baseline 12/16, baseline + auto-crop 14/16, model baru 12/16, model baru + auto-crop 12/16. Satu pisang overripe membaik, tetapi apel hijau dan sebagian jeruk masih salah; 16 gambar terlalu kecil untuk klaim statistik. Gap apel hijau/background masih perlu data tambahan.

File terkait: `models/mobilenetv3_external.keras`, `models/mobilenetv3_external_float32.tflite`, `reports/08_external_experiment.json`, `reports/09_wikimedia_external.csv`, dan skrip `notebooks/finetune_external.py`.
"""

code1 = """import json
import pandas as pd

report = json.loads((ROOT / "reports" / "08_external_experiment.json").read_text())
summary = report["summary"]
display(pd.DataFrame([
    {"model": "baseline", "test": "original", **{k: report["baseline"]["original_test"][k] for k in ("accuracy", "macro_f1")}},
    {"model": "finetuned", "test": "original", **{k: report["finetuned"]["original_test"][k] for k in ("accuracy", "macro_f1")}},
    {"model": "baseline", "test": "external", **{k: report["baseline"]["external_test"][k] for k in ("accuracy", "macro_f1")}},
    {"model": "finetuned", "test": "external", **{k: report["finetuned"]["external_test"][k] for k in ("accuracy", "macro_f1")}},
]).round(4))
display(pd.DataFrame(summary["train_counts"], index=["train"]).T)
print("Imbalance:", summary["imbalance_ratio_before"], "->", summary["imbalance_ratio_after"])
"""

markdown = {"cell_type": "markdown", "metadata": {}, "source": md1}
code = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code1}
# insert before final "Selesai" cell
idx = len(nb["cells"]) - 1
nb["cells"][idx:idx] = [markdown, code]

# Update final cell.
final = nb["cells"][-1]
final["source"] = """## Selesai

Seluruh pipeline 0–9 tersedia: EDA → praproses → CNN baseline → MobileNetV3 → perbandingan model → Grad-CAM + analisis defek → inferensi & TFLite → eksperimen data eksternal.

Model explainability/deployment terbaru: `models/mobilenetv3_external.keras` (**original test 99,60%**, external test 344 gambar **90,99%**). Baseline lama tetap tersedia di `models/mobilenetv3.keras`.
TFLite model baru: `models/mobilenetv3_external_float32.tflite` dan `models/mobilenetv3_external_dynamic.tflite`.
"""

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("updated", nb_path, "cells", len(nb["cells"]), "backup", backup)

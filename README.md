# FreshSight

Klasifikasi kesegaran apel, pisang, dan jeruk (6 kelas: fresh/rotten untuk masing-masing buah) menggunakan CNN baseline dan MobileNetV3Small.

## Struktur Project

```text
Citra + Viskom/
├── data/
│   ├── crops/                 # Crop per bounding box (generated, di-.gitignore)
│   ├── crops_external/        # Crop 224px dari data eksternal
│   └── external_test/         # 16 gambar uji eksternal + sources.csv
├── materials/
│   └── dataset/               # Dataset Pascal VOC + dataset eksternal (di-.gitignore)
│       ├── AFQC_apple/        # 2.320 apel fresh/rotten
│       ├── Banana_overripe/   # 1.195 banana overripe
│       └── orange_external/   # link dataset jeruk (belum diunduh)
├── models/
│   ├── cnn_baseline.keras
│   ├── mobilenetv3.keras
│   ├── mobilenetv3_external.keras   # hasil fine-tune data eksternal
│   ├── mobilenetv3_external_float32.tflite
│   ├── mobilenetv3_external_dynamic.tflite
│   ├── mobilenetv3_stage1.keras
│   ├── mobilenetv3_stage2.keras
│   ├── mobilenetv3_float32.tflite
│   └── mobilenetv3_dynamic.tflite
├── notebooks/
│   ├── FreshSight.ipynb       # Notebook utama: EDA, training, evaluasi, Grad-CAM, deployment
│   └── finetune_external.py   # Skrip eksperimen data eksternal
├── reports/
│   ├── 01_* ... 09_*          # Metrik, manifest, hasil evaluasi
│   └── figures/               # Visualisasi laporan
├── .gitignore
└── README.md
```

## File Utama

- **Notebook:** `notebooks/FreshSight.ipynb`
- **Dataset utama:** `materials/dataset/` (Pascal VOC: `JPEGImages/` + `Annotations/`)
- **Dataset eksternal:**
  - `materials/dataset/AFQC_apple/` — 2.320 citra apel fresh/rotten dari AFQC (lihat `SOURCE.md`)
  - `materials/dataset/Banana_overripe/` — hanya folder `over ripe` dari Banana_images_CNN (lihat `SOURCE.md`)
  - `materials/dataset/orange_external/` — link dataset jeruk Mendeley; unduh manual sesuai `SOURCE.md`
- **Model terbaik:** `models/mobilenetv3_external.keras`
- **Model deployment:** `models/mobilenetv3_external_float32.tflite` dan `models/mobilenetv3_external_dynamic.tflite`
- **Hasil evaluasi:** `reports/05_comparison.json`, `reports/08_external_experiment.json`
- **Uji eksternal:** `reports/07_external_test.csv`, `reports/08_external_test_predictions.csv`, `reports/09_wikimedia_external.csv`

## Ringkasan Hasil

| Model | Original test | Macro F1 | External test 344 | Ukuran |
|---|---:|---:|---:|---:|
| CNN baseline | 99,60% | 0,9906 | — | 5,16 MB |
| MobileNetV3 baseline | 99,20% | 0,9871 | 62,21% / F1 0,3223 | 11,84 MB |
| **MobileNetV3 + data eksternal** | **99,60%** | **0,9912** | **90,99% / F1 0,9071** | 10,05 MB |

External test 344 gambar: 116 fresh apple, 109 rotten apple, 119 rotten banana. Pada 16 gambar Wikimedia, model baru 12/16 tanpa auto-crop (baseline 12/16); auto-crop tidak selalu membantu. Gap apel hijau/background masih perlu data tambahan.

TFLite hasil fine-tune: float32 **99,60% original / 91,28% external**; dynamic **99,10% / 88,37%**. Detail: `reports/08_external_tflite_metrics.json`.

## Eksperimen Data Eksternal

- AFQC yang masuk training: 1.011 `fresh_apple` + 1.028 `rotten_apple` setelah dedupe.
- Banana yang masuk training: 954 `rotten_banana`; hanya folder `over ripe`.
- Test asli 997 objek tetap beku; external test 344 gambar.
- Imbalance turun dari **12,67×** menjadi **9,77×**; class weight dihitung ulang.
- Skrip: `notebooks/finetune_external.py`; evaluasi: `notebooks/evaluate_external_model.py`.

## Catatan

- File jurnal tidak disimpan di dalam project ini; versi jurnal ada di folder `../Jurnal/`.
- Backup file yang dihapus saat pembersihan tersedia di `../_backup_freshsight_2026-10-29.zip`.
- `data/crops/` dan `materials/dataset/` diabaikan oleh Git karena ukurannya besar/turunan.

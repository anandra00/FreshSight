# Progress FreshSight — 29 Oktober 2026

## Selesai

- Dataset utama Pascal VOC: 3.202 citra, 7.087 objek → setelah dedupe 3.093 citra / 6.977 objek, 6 kelas.
- CNN baseline: 99,60% accuracy, macro F1 0,9906.
- MobileNetV3 baseline: 99,20%, macro F1 0,9871.
- AFQC apple eksternal: 1.011 `fresh_apple` + 1.028 `rotten_apple` masuk training setelah dedupe.
- Banana overripe: hanya folder `over ripe`; 954 train, 119 val, 119 external test.
- Fine-tune MobileNetV3 dengan data eksternal:
  - Original test 997: **99,60%**, macro F1 **0,9912**.
  - External test 344 (apple/banana): **90,99%**, macro F1 **0,9071**.
  - Imbalance turun dari 12,67× menjadi 9,77×.
- TFLite:
  - float32: original 99,60%, external 91,28%.
  - dynamic: original 99,10%, external 88,37%.
- Notebook `FreshSight.ipynb` sudah memiliki Bagian 9 (eksperimen data eksternal).
- Laporan: `reports/08_external_experiment.json`, `reports/09_wikimedia_external.csv`.

## Belum Selesai / Ditunda

- Dataset jeruk 15.000 citra: download Mendeley sangat lambat dan sudah dihentikan; file parsial dibersihkan. Link dan mapping ada di `materials/dataset/orange_external/SOURCE.md`.
- Green apple/rotten green apple: AFVC Granny Smith hanya tersedia untuk apel hijau fresh; belum ada dataset rotten green apple yang tervalidasi.
- Background sintetis + real (piring/meja/keranjang): belum diimplementasikan.
- Pada 16 gambar Wikimedia, model baru masih 12/16; auto-crop tidak selalu membantu. Gap apel hijau/background dan jeruk masih terbuka.

## Catatan

- Test asli 997 objek tidak diubah dan tetap menjadi evaluasi utama.
- Data eksternal hanya masuk train/val.
- Model terbaru: `models/mobilenetv3_external.keras`.
- Model baseline tetap tersedia: `models/mobilenetv3.keras`.
- Detail struktur dan cara menjalankan ada di `README.md`.

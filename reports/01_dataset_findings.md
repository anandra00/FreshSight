# Ringkasan Dataset — FreshSight

- Lokasi: `C:\Users\anand\OneDrive\Documents\Semester 5\Pemrosesan CItra\PJBL\Citra + Viskom\materials\dataset`
- Struktur: **Pascal VOC (object detection): JPEGImages/ + Annotations/*.xml**
- Gambar: **3202** (3202 valid, 109 grup duplikat persis)
- Objek beranotasi: **7087** → setelah dedupe: **6977**
- Gambar berisi >1 label: **1308**
- Rentang ukuran: 416–2104 px (median 640 px)
- Near-duplicate (dHash ≤ 2): **1379 klaster**, 2354 gambar
  (76.1%) berada di klaster berisi >1 gambar; terbesar 43 gambar.

## Distribusi kelas (objek)
- `fresh_apple`: 705
- `normal_apple`: 1103
- `rotten_apple`: 179
- `fresh_banana`: 904
- `normal_banana`: 1356
- `rotten_banana`: 781
- `fresh_orange`: 737
- `normal_orange`: 995
- `rotten_orange`: 327

## Distribusi 6 kelas yang dipakai (normal -> fresh)
- `fresh_apple`: 1726
- `rotten_apple`: 178
- `fresh_banana`: 2248
- `rotten_banana`: 778
- `fresh_orange`: 1724
- `rotten_orange`: 323

## Kebocoran klaster per strategi split
- Naif per objek: **546** klaster bocor
- Acak per gambar: **379** klaster bocor
- Per klaster (dipakai): **0** klaster bocor

## Keputusan
1. **Format data**: crop per bounding box — dipakai.
2. **Label**: **6 kelas** (`normal` → `fresh`) — diputuskan.
3. **Duplikat**: 109 grup duplikat persis dibuang; near-duplicate dikelompokkan
   agar tidak tersebar antar split.
4. **Split**: per klaster near-duplicate + stratifikasi label dominan, 70/15/15.

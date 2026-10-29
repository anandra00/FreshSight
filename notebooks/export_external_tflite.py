import tensorflow as tf
from pathlib import Path

ROOT = Path(r"C:\Users\anand\OneDrive\Documents\Semester 5\Pemrosesan CItra\PJBL\Citra + Viskom")
model = tf.keras.models.load_model(ROOT / "models" / "mobilenetv3_external.keras", compile=False)
export_dir = ROOT / "models" / "savedmodel_mobilenetv3_external"
if not export_dir.exists():
    model.export(str(export_dir))

for name, dynamic in [("mobilenetv3_external_float32.tflite", False),
                      ("mobilenetv3_external_dynamic.tflite", True)]:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    if dynamic:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    data = converter.convert()
    out = ROOT / "models" / name
    out.write_bytes(data)
    print(name, f"{out.stat().st_size/1e6:.2f} MB")

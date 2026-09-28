import sys
import cv2
import numpy as np
from keras.models import load_model


# Model file yang dihasilkan oleh freshness_regression_adapted.ipynb
MODEL_PATH = 'freshness_model_final.h5'

# Thresholds untuk 3-level freshness classification
# Score model: 0.0 (fresh) - 0.5 (normal) - 1.0 (rotten)
THRESHOLD_FRESH = 0.33
THRESHOLD_MEDIUM = 0.66


def classify_freshness(score):
    """
    Klasifikasi freshness berdasarkan prediction score.
    Score < 0.33 -> FRESH
    Score 0.33 - 0.66 -> MEDIUM FRESH
    Score >= 0.66 -> NOT FRESH (ROTTEN)
    """
    if score < THRESHOLD_FRESH:
        return "FRESH"
    elif score < THRESHOLD_MEDIUM:
        return "MEDIUM FRESH"
    else:
        return "NOT FRESH"


def pre_proc_img(image_path):
    """Baca dan preprocess gambar untuk prediksi model."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Tidak bisa membaca gambar '{image_path}'")
        sys.exit(1)

    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize ke [0, 1]
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def evaluate_freshness(image_path, model_path=MODEL_PATH):
    """
    Evaluasi freshness dari sebuah gambar.
    Returns: prediction score (float, 0-1)
    """
    model = load_model(model_path)
    prediction = model.predict(pre_proc_img(image_path), verbose=0)
    return prediction[0][0]


if __name__ == '__main__':
    # Ambil path gambar dari argumen atau gunakan default
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = 'image-to-eval.png'

    print(f"Evaluating: {img_path}")
    print(f"Model: {MODEL_PATH}")
    print("-" * 40)

    score = evaluate_freshness(img_path)
    classification = classify_freshness(score)

    print(f"Prediction Score: {score:.4f}")
    print(f"Classification: {classification}")

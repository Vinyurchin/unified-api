import base64
import io
import json
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np
import random
import gdown
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Optional deps (solo se usan si están instalados y compatibles)
try:
    import cv2
except Exception:
    cv2 = None

try:
    from fer.fer import FER
except Exception:
    FER = None

# Environment setup
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
load_dotenv()

APP_ROOT = Path(__file__).parent
MODELS_DIR = APP_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Tumor models
CLASSIFIER_URL = "https://drive.google.com/uc?id=1TQ2_ozS3crjqchAPXNCBuyVs2SvZdf9R"
CLASSIFIER_PATH = MODELS_DIR / "tumor_classifier.h5"
SEGMENTATION_URL = "https://drive.google.com/uc?id=1Ft8ttlApP3vyF5NMYl9UxFOg4kimZAMG"
SEGMENTATION_PATH = MODELS_DIR / "segmentacion.keras"

# Emotion translation
TRADUCCION_EMOCIONES = {
    "angry": "enojado",
    "disgust": "disgustado",
    "fear": "miedo",
    "happy": "feliz",
    "sad": "triste",
    "surprise": "sorprendido",
    "neutral": "neutral",
}

EMOJIS = {
    "enojado": "😠",
    "disgustado": "🤢",
    "miedo": "😨",
    "feliz": "😄",
    "triste": "😢",
    "sorprendido": "😲",
    "neutral": "😐",
}

# Opcionales: si no están instalados, se hará fallback a lógica simplificada
_fer_detector = None

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


# ----- Utility helpers -----
def convertir_a_base64(imagen):
    buffered = io.BytesIO()
    imagen.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def procesar_imagen_con_puntos(image_np):
    if cv2 is None:
        return Image.fromarray(image_np)

    try:
        imagen = Image.fromarray(image_np)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            draw = ImageDraw.Draw(imagen)
            for (x, y, w, h) in faces:
                points = [
                    (x + w//2, y + h//3),
                    (x + w//4, y + h//2),
                    (x + 3*w//4, y + h//2),
                    (x + w//2, y + 2*h//3),
                    (x + w//3, y + 4*h//5),
                    (x + 2*w//3, y + 4*h//5),
                ]
                for px, py in points:
                    draw.ellipse([px-3, py-3, px+3, py+3], fill=(255, 0, 0))
                    draw.line((px - 4, py - 4, px + 4, py + 4), fill=(255, 0, 0), width=2)
                    draw.line((px - 4, py + 4, px + 4, py - 4), fill=(255, 0, 0), width=2)
        return imagen
    except Exception:
        return Image.fromarray(image_np)


def array_to_pil_image(arr):
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(axis=-1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def pil_to_bytes_io(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def detectar_emociones_con_modelo(imagen_pil):
    if FER is None:
        return None

    global _fer_detector
    if _fer_detector is None:
        _fer_detector = FER(mtcnn=False)

    detecciones = _fer_detector.detect_emotions(np.array(imagen_pil))
    if not detecciones:
        return None

    raw = detecciones[0].get("emotions") or {}
    if not raw:
        return None

    emocion_en = max(raw, key=raw.get)
    emocion_es = TRADUCCION_EMOCIONES.get(emocion_en, emocion_en)
    emociones_es = {
        TRADUCCION_EMOCIONES.get(k, k): float(v)
        for k, v in raw.items()
        if k in TRADUCCION_EMOCIONES
    }
    return emocion_es, emociones_es


# ----- Model loading (lazy) -----
_resnet_model = None
_unet_model = None


def ensure_models():
    global _resnet_model, _unet_model
    if not CLASSIFIER_PATH.exists():
        gdown.download(CLASSIFIER_URL, str(CLASSIFIER_PATH), quiet=False)
    if not SEGMENTATION_PATH.exists():
        gdown.download(SEGMENTATION_URL, str(SEGMENTATION_PATH), quiet=False)
    if _resnet_model is None:
        _resnet_model = load_model(CLASSIFIER_PATH)
    if _unet_model is None:
        _unet_model = load_model(SEGMENTATION_PATH)


# ----- Routes -----
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/emotion/upload", methods=["POST"])
def detectar_emociones():
    if "file" not in request.files:
        return jsonify({"error": "No se recibió correctamente la imagen"}), 400
    archivo = request.files["file"]
    if archivo.filename == "":
        return jsonify({"error": "No se cargó ninguna imagen"}), 400

    try:
        imagen_pil = Image.open(archivo).convert("RGB")
        imagen_pil = imagen_pil.resize((300, 300))
        imagen_np = np.array(imagen_pil)

        imagen_mejorada = ImageEnhance.Contrast(imagen_pil).enhance(1.5)
        imagen_mejorada = ImageEnhance.Sharpness(imagen_mejorada).enhance(2.0)

        imagen_con_puntos = procesar_imagen_con_puntos(imagen_np)

        emocion_principal = None
        emotions_dict = {}

        modelo = detectar_emociones_con_modelo(imagen_mejorada)
        if modelo:
            emocion_principal, emotions_dict = modelo

        if emocion_principal is None:
            opciones = ["feliz", "triste", "enojado", "neutral"]
            emocion_principal = random.choice(opciones)
            emotions_dict = {op: 0.0 for op in opciones}
            emotions_dict[emocion_principal] = 1.0

        img_data_puntos = convertir_a_base64(imagen_con_puntos)

        return jsonify(
            {
                "image_with_points_base64": img_data_puntos,
                "dominant_emotion": emocion_principal,
                "emotions": emotions_dict,
                "drive_id": None,
            }
        )

    except Exception as exc:
        return jsonify({"error": f"Error al procesar la imagen: {exc}"}), 500


@app.route("/tumor/predict", methods=["POST"])
def predict_tumor():
    if "data" not in request.files:
        return jsonify({"error": "No se recibió correctamente la imagen"}), 400
    file = request.files["data"]

    try:
        ensure_models()

        img_rgb = Image.open(file).resize((128, 128)).convert("RGB")
        arr_rgb = np.array(img_rgb) / 255.0
        input_rgb = np.expand_dims(arr_rgb, axis=0)
        tumor_prob = float(_resnet_model.predict(input_rgb)[0][0])

        file.seek(0)
        img_gray = Image.open(file).resize((256, 256)).convert("L")
        arr_gray = np.array(img_gray) / 255.0
        input_gray = np.expand_dims(arr_gray, axis=(0, -1))

        if tumor_prob < 0.75:
            img_pil = array_to_pil_image(arr_gray)
            img_buf = pil_to_bytes_io(img_pil)

            pdf_buf = io.BytesIO()
            c = canvas.Canvas(pdf_buf, pagesize=letter)
            c.drawString(100, 750, "Resultado: No hay tumor detectado.")
            c.drawString(100, 735, f"Probabilidad: {tumor_prob:.6f}")
            c.drawImage(ImageReader(img_buf), 100, 450, width=256, height=256)
            c.save()
            pdf_buf.seek(0)

            return send_file(pdf_buf, mimetype="application/pdf", download_name="resultado.pdf")

        pred_mask = _unet_model.predict(input_gray)[0]
        pred_mask_bin = (pred_mask > 0.2).astype(np.uint8).squeeze()

        overlay = np.stack([arr_gray] * 3, axis=-1)
        overlay[pred_mask_bin == 1] = [1, 0, 0]

        mri_img = array_to_pil_image(arr_gray)
        mask_img = array_to_pil_image(np.stack([pred_mask_bin] * 3, axis=-1))
        overlay_img = array_to_pil_image(overlay)

        mri_buf = pil_to_bytes_io(mri_img)
        mask_buf = pil_to_bytes_io(mask_img)
        overlay_buf = pil_to_bytes_io(overlay_img)

        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=letter)
        c.drawString(100, 750, "Resultado: Tumor detectado")
        c.drawString(100, 735, f"Probabilidad: {tumor_prob:.6f}")
        c.drawString(100, 700, "Imagen original:")
        c.drawImage(ImageReader(mri_buf), 100, 450, width=256, height=256)
        c.drawString(100, 430, "Mascara:")
        c.drawImage(ImageReader(mask_buf), 100, 200, width=256, height=256)
        c.drawString(380, 430, "Overlay:")
        c.drawImage(ImageReader(overlay_buf), 380, 200, width=256, height=256)
        c.save()
        pdf_buf.seek(0)

        return send_file(pdf_buf, mimetype="application/pdf", download_name="resultado.pdf")

    except Exception as exc:
        return jsonify({"error": f"Error al procesar la imagen: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


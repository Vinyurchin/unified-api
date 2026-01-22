import base64
import io
import os
from pathlib import Path

# Environment setup BEFORE importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_USE_LEGACY_KERAS"] = "True"

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np
import gdown
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from dotenv import load_dotenv

# Optional deps (solo se usan si están instalados y compatibles)
try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from fer.fer import FER
    _fer_detector_instance = None
except Exception:
    FER = None
    _fer_detector_instance = None

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

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


# ----- Utility helpers -----
def convertir_a_base64(imagen):
    buffered = io.BytesIO()
    imagen.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def procesar_imagen_con_puntos_mediapipe(image_np):
    """Procesa la imagen y añade puntos faciales usando MediaPipe (más preciso)."""
    if mp is None:
        return Image.fromarray(image_np)
    
    try:
        imagen = Image.fromarray(image_np)
        mp_face_mesh = mp.solutions.face_mesh
        puntos_deseados = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(image_np)
            if results.multi_face_landmarks:
                draw = ImageDraw.Draw(imagen)
                for face_landmarks in results.multi_face_landmarks:
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        if idx in puntos_deseados:
                            h, w, _ = image_np.shape
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0), width=2)
                            draw.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 0, 0), width=2)
        return imagen
    except Exception:
        return Image.fromarray(image_np)


def procesar_imagen_con_puntos_opencv(image_np):
    """Fallback: Procesa la imagen con OpenCV (menos preciso)."""
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


def procesar_imagen_con_puntos(image_np):
    """Intenta MediaPipe primero, luego fallback a OpenCV."""
    # Intentar con MediaPipe (más preciso)
    if mp is not None:
        resultado = procesar_imagen_con_puntos_mediapipe(image_np)
        if resultado:
            return resultado
    
    # Fallback a OpenCV
    return procesar_imagen_con_puntos_opencv(image_np)


def get_fer_detector():
    """Obtiene el detector FER de forma lazy y segura."""
    global _fer_detector_instance
    if FER is None:
        return None
    
    if _fer_detector_instance is None:
        try:
            _fer_detector_instance = FER(mtcnn=False)
        except Exception as e:
            print(f"Advertencia: Error al inicializar FER: {e}")
            return None
    
    return _fer_detector_instance


def detectar_emociones_con_fer(imagen_pil, imagen_np):
    """Intenta detectar emociones con FER - usa detección real."""
    detector = get_fer_detector()
    if detector is None:
        print("[FER] Detector no disponible")
        return None
    
    try:
        # FER funciona mejor con RGB en formato uint8
        if imagen_np.dtype != np.uint8:
            if imagen_np.max() <= 1.0:
                imagen_np = (imagen_np * 255).astype(np.uint8)
            else:
                imagen_np = imagen_np.astype(np.uint8)
        
        print(f"[FER] Analizando imagen: shape={imagen_np.shape}, dtype={imagen_np.dtype}")
        
        # Detectar emociones
        emociones = detector.detect_emotions(imagen_np)
        
        if not emociones or len(emociones) == 0:
            print("[FER] No se detectaron rostros")
            return None
        
        # Obtener la primera cara detectada
        rostro_data = emociones[0]
        emociones_detectadas = rostro_data.get("emotions", {})
        
        if not emociones_detectadas:
            print("[FER] Rostro detectado pero sin emociones")
            return None
        
        print(f"[FER] Emociones detectadas: {emociones_detectadas}")
        
        # Obtener emoción principal
        emocion_principal_en = max(emociones_detectadas, key=emociones_detectadas.get)
        emocion_principal = TRADUCCION_EMOCIONES.get(emocion_principal_en, emocion_principal_en)
        
        # Convertir emociones al español
        emociones_es = {
            TRADUCCION_EMOCIONES.get(emocion_en, emocion_en): round(float(valor), 4)
            for emocion_en, valor in emociones_detectadas.items()
        }
        
        print(f"[FER] ✓ Emoción principal: {emocion_principal}")
        return emocion_principal, emociones_es
    
    except Exception as e:
        print(f"[FER] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def detectar_emociones_fallback_opencv(imagen_np):
    """Fallback: Detección con OpenCV + emociones simuladas."""
    if cv2 is None:
        return None

    try:
        gray = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None

        # Retornar emociones simuladas (distribución neutral)
        return "neutral", {
            "enojado": 0.08,
            "disgustado": 0.05,
            "miedo": 0.06,
            "feliz": 0.22,
            "triste": 0.09,
            "sorprendido": 0.12,
            "neutral": 0.38
        }
    except Exception as e:
        print(f"[OpenCV] Error: {e}")
        return None



def array_to_pil_image(arr):
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(axis=-1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def pil_to_bytes_io(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf



# ----- Model loading (lazy) -----
_resnet_model = None
_unet_model = None
_input_layer_patched = False


def patch_input_layer_batch_shape():
    """Permite cargar modelos antiguos que usan 'batch_shape' en InputLayer."""
    global _input_layer_patched
    if _input_layer_patched:
        return

    try:
        original_init = InputLayer.__init__

        def patched_init(self, *args, **kwargs):
            if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
            return original_init(self, *args, **kwargs)

        InputLayer.__init__ = patched_init
        _input_layer_patched = True
        print("[Patch] InputLayer admite 'batch_shape' -> 'batch_input_shape'")
    except Exception as exc:
        print(f"[Patch] No se pudo parchear InputLayer: {exc}")


def ensure_models():
    global _resnet_model, _unet_model
    if not CLASSIFIER_PATH.exists():
        gdown.download(CLASSIFIER_URL, str(CLASSIFIER_PATH), quiet=False)
    if not SEGMENTATION_PATH.exists():
        gdown.download(SEGMENTATION_URL, str(SEGMENTATION_PATH), quiet=False)
    patch_input_layer_batch_shape()
    
    if _resnet_model is None:
        print("[Model] Cargando tumor_classifier.h5...")
        try:
            from tensorflow.keras.utils import custom_object_scope
            with custom_object_scope({}):
                _resnet_model = load_model(CLASSIFIER_PATH, compile=False)
        except Exception as e:
            print(f"[Model] Error al cargar con custom_object_scope: {e}")
            _resnet_model = load_model(CLASSIFIER_PATH, compile=False)
    
    if _unet_model is None:
        print("[Model] Cargando segmentacion.keras...")
        try:
            from tensorflow.keras.utils import custom_object_scope
            with custom_object_scope({}):
                _unet_model = load_model(SEGMENTATION_PATH, compile=False)
        except Exception as e:
            print(f"[Model] Error al cargar con custom_object_scope: {e}")
            try:
                _unet_model = load_model(SEGMENTATION_PATH, compile=False, safe_mode=False)
            except Exception as e2:
                print(f"[Model] Error con safe_mode=False: {e2}")
                _unet_model = load_model(SEGMENTATION_PATH, compile=False)


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
        # Leer imagen original
        imagen_pil = Image.open(archivo).convert("RGB")
        # NO reducir tanto - mantener más detalles para FER
        imagen_pil = imagen_pil.resize((640, 640))
        imagen_np = np.array(imagen_pil)

        # Mejorar imagen para mejor detección
        imagen_mejorada = ImageEnhance.Contrast(imagen_pil).enhance(1.2)
        imagen_mejorada = ImageEnhance.Sharpness(imagen_mejorada).enhance(1.5)
        imagen_mejorada_np = np.array(imagen_mejorada)

        print(f"[ENDPOINT] Procesando imagen de tamaño {imagen_np.shape}")

        emocion_principal = None
        emotions_dict = {}
        metodo_usado = "desconocido"

        # Intentar FER primero (más preciso)
        print("[ENDPOINT] Intentando detección con FER...")
        resultado_fer = detectar_emociones_con_fer(imagen_pil, imagen_mejorada_np)
        
        if resultado_fer:
            print("[ENDPOINT] ✓ FER detectó emociones correctamente")
            emocion_principal, emotions_dict = resultado_fer
            metodo_usado = "FER"
        else:
            # Fallback a OpenCV
            print("[ENDPOINT] FER sin resultados, intentando OpenCV...")
            resultado_fallback = detectar_emociones_fallback_opencv(imagen_mejorada_np)
            
            if resultado_fallback:
                print("[ENDPOINT] ✓ OpenCV detectó rostro")
                emocion_principal, emotions_dict = resultado_fallback
                metodo_usado = "OpenCV+Simulado"
            else:
                print("[ENDPOINT] ✗ No se detectó rostro")
                return jsonify({
                    "error": "No se detectó rostro en la imagen. Intenta con:",
                    "sugerencias": [
                        "Una imagen más clara",
                        "Mejor iluminación",
                        "Rostro mirando hacia la cámara",
                        "Sin obstáculos en el rostro"
                    ]
                }), 400

        # Si llegamos aquí, tenemos detección exitosa
        print(f"[ENDPOINT] ✓ Detección exitosa con {metodo_usado}: {emocion_principal}")
        
        # Procesar puntos faciales para visualización (imagen pequeña para render)
        imagen_display = imagen_pil.resize((300, 300))
        imagen_display_np = np.array(imagen_display)
        imagen_con_puntos = procesar_imagen_con_puntos(imagen_display_np)

        img_data_puntos = convertir_a_base64(imagen_con_puntos)

        response_data = {
            "image_with_points_base64": img_data_puntos,
            "dominant_emotion": emocion_principal,
            "emotions": emotions_dict,
            "detection_method": metodo_usado
        }
        
        print(f"[ENDPOINT] ✓ Respuesta enviada correctamente")
        return jsonify(response_data), 200

    except Exception as exc:
        print(f"[ENDPOINT] ✗ Error en detección: {exc}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error al procesar la imagen: {str(exc)}"}), 500


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


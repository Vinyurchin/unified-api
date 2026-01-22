# Unified API - Análisis Médico IA

Sistema integrado para detección de emociones en rostros y análisis de tumores en imágenes MRI.

## ✨ Características

- **Detección de Emociones**: Análisis de 7 emociones básicas usando FER (Facial Expression Recognition)
- **Puntos Faciales**: Detección de landmarks faciales usando MediaPipe
- **Análisis de Tumores MRI**: Clasificación y segmentación de tumores cerebrales
- **Generación de PDFs**: Reportes automáticos con análisis y visualizaciones
- **Frontend Web**: Interfaz completa para ambas funcionalidades

## 🎯 Emociones Detectables

1. Enojado (😠)
2. Disgustado (🤢)
3. Miedo (😨)
4. Feliz (😄)
5. Triste (😢)
6. Sorprendido (😲)
7. Neutral (😐)

## 🚀 Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/Scripts/activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 📦 Dependencias Principales

- Flask 2.0.1
- TensorFlow (CPU)
- MediaPipe - Landmarks faciales
- FER - Detección de emociones
- OpenCV - Visión por computadora
- ReportLab - Generación de PDFs
- Pillow - Procesamiento de imágenes

## ▶️ Uso

```bash
python app.py
```

El servidor estará disponible en `http://localhost:5000`

## 🔌 Endpoints API

### Detección de Emociones
- **POST** `/emotion/upload`
- Parámetro: `file` (imagen)
- Respuesta: Emoción detectada + puntos faciales en base64

### Análisis de Tumores
- **POST** `/tumor/predict`
- Parámetro: `data` (imagen MRI)
- Respuesta: PDF con análisis, máscara y overlay

### Frontend
- **GET** `/`
- Interfaz web interactiva

## 📝 Estructura del Código

```
app.py
├── Imports y configuración
├── Funciones de procesamiento
│   ├── Detección de puntos faciales (MediaPipe + OpenCV)
│   ├── Detección de emociones (FER)
│   └── Conversión de imágenes
├── Carga de modelos (lazy loading)
└── Rutas y endpoints
```

## ⚙️ Configuración

- Tamaño de imagen emociones: 640x640 (detección) → 300x300 (display)
- Tamaño de imagen tumores: 128x128 (RGB), 256x256 (escala gris)
- Umbral tumor: 0.75
- Umbral segmentación: 0.2

## 📊 Modelos IA

Los modelos se descargan automáticamente en la primera ejecución desde Google Drive:

- `tumor_classifier.h5` - ResNet para clasificación de tumores
- `segmentacion.keras` - UNet para segmentación

## 🔧 Desarrollo

### Agregar nuevas funcionalidades
1. Crear función en el archivo principal
2. Documentar con docstrings
3. Agregar logs con prefijos `[MODULO]`
4. Testear antes de hacer commit

### Estructura de logs
```
[FER] - Detección de emociones
[OpenCV] - Fallbacks visuales
[ENDPOINT] - Rutas Flask
```

## 📈 Performance

- FER: ~2-3 segundos por rostro
- Tumor: ~5-10 segundos (descarga de modelos en primera ejecución)
- Puntos faciales: <1 segundo

## 🐛 Troubleshooting

**No se detectan emociones:**
- Verificar iluminación de la imagen
- Asegurar que el rostro esté mirando hacia la cámara
- Usar imágenes claras sin obstáculos

**Errores con modelos:**
- Verificar conexión a internet (descarga de modelos)
- Limpiar caché de gdown si hay corrupción

## 📄 Licencia

Proyecto académico para análisis médico e interpretación de emociones.

## 👥 Autor

Vinyurchin - 2026

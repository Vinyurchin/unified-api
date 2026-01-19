# 🔬 Análisis Médico IA - Detección de Emociones y Tumores

Backend unificado que combina dos modelos de IA:
- **Detección de Emociones**: Análisis facial con puntos de referencia usando MediaPipe y FER
- **Detección de Tumores**: Clasificación y segmentación de MRI cerebrales con TensorFlow

## Endpoints

### POST `/emotion/upload`
Detecta emociones en imágenes faciales.

**Request:**
```
Content-Type: multipart/form-data
file: <imagen>
```

**Response:**
```json
{
  "dominant_emotion": "feliz",
  "emotions": {"angry": 0.05, "happy": 0.85, ...},
  "image_with_points_base64": "...",
  "drive_id": null
}
```

### POST `/tumor/predict`
Analiza MRI para detectar tumores cerebrales.

**Request:**
```
Content-Type: multipart/form-data
data: <imagen_mri>
```

**Response:**
PDF con análisis visual (imagen original, máscara, overlay)

## Despliegue en Heroku

### Prerequisitos
- Heroku CLI instalado
- Cuenta de Heroku
- Git configurado

### Pasos

1. **Crear app en Heroku:**
```bash
heroku create tu-app-name
```

2. **Configurar stack a container:**
```bash
heroku stack:set container -a tu-app-name
```

3. **Añadir variables de entorno (opcional - solo si usas Google Drive):**
```bash
heroku config:set GOOGLE_DRIVE_CREDENTIALS='{"type": "service_account", ...}' -a tu-app-name
heroku config:set FOLDER_ID='tu_folder_id' -a tu-app-name
```

4. **Desplegar:**
```bash
git push heroku main
```

5. **Abrir app:**
```bash
heroku open -a tu-app-name
```

## Variables de Entorno

- `GOOGLE_DRIVE_CREDENTIALS` (opcional): JSON completo de cuenta de servicio para subir emociones a Drive
- `FOLDER_ID` (opcional): ID de carpeta en Drive donde subir imágenes

Si no se configuran, la app funciona sin Drive (recomendado para comenzar).

## Desarrollo Local

```bash
pip install -r requirements.txt
python app.py
```

Luego accede a `http://localhost:5000/`

## Estructura

```
unified-api/
├── app.py              # Flask app con ambos endpoints
├── static/
│   └── index.html      # Frontend unificado
├── models/             # Modelos descargados en runtime
├── requirements.txt
├── Dockerfile
├── Procfile
└── README.md
```

## Notas

- Los modelos de tumor se descargan automáticamente en el primer arranque (~110 MB)
- Timeout: 300 segundos para análisis largos
- Usar dyno Hobby ($7) o superior para evitar OOM

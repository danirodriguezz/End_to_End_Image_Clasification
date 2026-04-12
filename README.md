# End-to-End Image Classifier

Clasifica imágenes en 4 categorías: **airplane · bicycle · car · dog**

```
[Upload imagen] → [FastAPI /predict] → [ResNet18] → [% confianza por clase]
```

---

## Estructura del proyecto

```
.
├── config.py               # Hiperparámetros y rutas centralizados
├── train_pipeline.py       # Script principal de entrenamiento
├── requirements.txt        # Dependencias
│
├── src/
│   ├── dataset.py          # CIFAR-10 + CIFAR-100 → 4 clases
│   ├── model.py            # ResNet18 con cabeza personalizada
│   ├── train.py            # Bucle de entrenamiento y validación
│   └── transforms.py       # Augmentaciones y preprocesado
│
├── api/
│   ├── main.py             # FastAPI app + endpoint /predict
│   └── inference.py        # Carga del modelo e inferencia
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── data/                   # CIFAR descargado automáticamente
└── models/                 # Pesos guardados tras el entrenamiento
    ├── best_model_weights.pth
    └── model_metadata.json
```

---

## Instalación

```bash
# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Paso 1 — Entrenar el modelo

```bash
python train_pipeline.py
```

- Descarga CIFAR-10 y CIFAR-100 automáticamente en `data/`
- Filtra 4 clases: airplane (CIFAR-10), bicycle (CIFAR-100), car (CIFAR-10), dog (CIFAR-10)
- Usa **ResNet18** pre-entrenado en ImageNet con fine-tuning en dos fases:
  - **Fase 1** (épocas 1-5): backbone congelado, solo entrena la cabeza FC
  - **Fase 2** (épocas 6-20): fine-tuning completo con learning rates diferenciados
- Guarda el mejor modelo en `models/best_model_weights.pth`
- Guarda metadatos en `models/model_metadata.json`

Tiempo estimado: ~10 min en CPU, ~2 min en GPU.

---

## Paso 2 — Arrancar el backend

```bash
uvicorn api.main:app --reload --port 8000
```

La API queda disponible en `http://localhost:8000`

### Endpoints

| Método | Ruta        | Descripción                          |
|--------|-------------|--------------------------------------|
| GET    | `/health`   | Comprobación de estado               |
| GET    | `/classes`  | Lista de clases soportadas           |
| POST   | `/predict`  | Clasifica una imagen subida          |
| GET    | `/`         | Interfaz web (frontend estático)     |
| GET    | `/docs`     | Documentación interactiva (Swagger)  |

### Ejemplo con curl

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@foto_perro.jpg"
```

Respuesta:
```json
{
  "predictions": [
    {"class": "dog",      "confidence": 0.9241},
    {"class": "car",      "confidence": 0.0432},
    {"class": "bicycle",  "confidence": 0.0201},
    {"class": "airplane", "confidence": 0.0126}
  ],
  "top_class": "dog"
}
```

---

## Paso 3 — Usar la interfaz web

Abre el navegador en:

```
http://localhost:8000
```

- Arrastra una imagen o haz clic para seleccionarla
- El modelo responde con barras de confianza para cada clase

---

## Arquitectura del modelo

```
ResNet18 (ImageNet pre-trained)
  └── FC: 512 → 4 (airplane, bicycle, car, dog)
```

**Técnicas aplicadas:**
- Transfer learning desde ImageNet
- Fine-tuning en dos fases con learning rates diferenciados
- WeightedRandomSampler para compensar el desbalance de clases (bicycle tiene 10× menos datos)
- CrossEntropyLoss con pesos por clase
- Data augmentation: flip horizontal, rotación, ColorJitter, RandomAffine

---

## Dataset

| Clase    | Fuente      | Train | Val  |
|----------|-------------|-------|------|
| airplane | CIFAR-10    | 5 000 | 1 000|
| bicycle  | CIFAR-100   |   500 |   100|
| car      | CIFAR-10    | 5 000 | 1 000|
| dog      | CIFAR-10    | 5 000 | 1 000|

---

## Notas técnicas

- Las imágenes CIFAR (32×32) se reescalan a 224×224 con `transforms.Resize`
- Se usa normalización ImageNet porque los pesos iniciales son de ImageNet
- El modelo se carga una sola vez al arrancar el servidor (no por petición)
- Las imágenes PNG con canal alfa (RGBA) se convierten a RGB automáticamente

---
title: Image Classifier ResNet18
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# Image Classifier

**Clasificación de imágenes en tiempo real con ResNet18 + FastAPI**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-99.26%25-22c55e?style=flat-square)

Sube una imagen y obtén predicciones con nivel de confianza para 4 clases: **avión · bicicleta · coche · perro**

</div>

---

## Cómo funciona

```
Imagen subida
     │
     ▼
POST /predict  (FastAPI)
     │
     ▼
PIL → Resize 224×224 → Normalización ImageNet
     │
     ▼
ResNet18 fine-tuned  (512 → 4 logits)
     │
     ▼
Softmax → ordenado por confianza
     │
     ▼
{ "top_class": "dog", "predictions": [...] }
```

---

## Resultados

| Clase    | Precisión | Recall | F1-score | Muestras val |
|----------|:---------:|:------:|:--------:|:------------:|
| airplane |   0.99    |  1.00  |   1.00   |    1 000     |
| bicycle  |   1.00    |  0.95  |   0.97   |      100     |
| car      |   1.00    |  0.99  |   1.00   |    1 000     |
| dog      |   0.99    |  1.00  |   1.00   |    1 000     |
| **avg**  | **0.995** | **0.985** | **0.99** | **3 100** |

> Accuracy de validación: **99.26 %**

---

## Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| Modelo | ResNet18 pre-entrenado (ImageNet) + fine-tuning |
| Training | PyTorch · torchvision · scikit-learn |
| Backend | FastAPI · Uvicorn |
| Frontend | HTML + CSS + JavaScript vanilla |
| Datos | CIFAR-10 + CIFAR-100 (descarga automática) |

---

## Instalación y uso

### 1. Preparar el entorno

```bash
git clone https://github.com/tu-usuario/End_to_End_Image_Clasifications.git
cd End_to_End_Image_Clasifications

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Entrenar el modelo

```bash
python train_pipeline.py
```

Descarga CIFAR-10 y CIFAR-100 automáticamente, entrena durante 20 épocas y guarda los mejores pesos en `models/`.

> ~10 min en CPU · ~2 min en GPU

### 3. Arrancar el servidor

```bash
uvicorn api.main:app --reload --port 8000
```

| URL | Descripción |
|-----|-------------|
| http://localhost:8000 | Interfaz web |
| http://localhost:8000/docs | Swagger UI |

### 4. Predecir desde la línea de comandos

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@mi_imagen.jpg"
```

```json
{
  "predictions": [
    { "class": "dog",      "confidence": 0.9241 },
    { "class": "car",      "confidence": 0.0432 },
    { "class": "bicycle",  "confidence": 0.0201 },
    { "class": "airplane", "confidence": 0.0126 }
  ],
  "top_class": "dog"
}
```

---

## Arquitectura del modelo

```
ResNet18 (ImageNet pre-trained)
  ├── Conv1 → BN → ReLU → MaxPool
  ├── Layer1-4  (residual blocks, congelados en fase 1)
  └── FC: 512 → 4  ← cabeza nueva (siempre entrenable)
```

### Estrategia de entrenamiento en dos fases

| Fase | Épocas | Capas entrenables | LR cabeza | LR backbone |
|------|:------:|:-----------------:|:---------:|:-----------:|
| 1 — warmup | 1–5   | Solo FC head | 1e-3 | — |
| 2 — fine-tuning | 6–20 | Toda la red | 1e-3 | 1e-4 |

La fase 1 estabiliza la nueva cabeza antes de propagar gradientes por el backbone, evitando destruir los features aprendidos de ImageNet.

### Gestión del desbalance de clases

Bicycle tiene 10× menos muestras que el resto (500 vs 5 000):

- **`WeightedRandomSampler`** — oversampling de bicicleta en cada batch
- **`CrossEntropyLoss(weight=[1, 10, 1, 1])`** — mayor penalización por errores en bicicleta

---

## Dataset

| Clase    | Fuente    | Índice original | Train | Val   |
|----------|-----------|:---------------:|------:|------:|
| airplane | CIFAR-10  | 0               | 5 000 | 1 000 |
| bicycle  | CIFAR-100 | 8               |   500 |   100 |
| car      | CIFAR-10  | 1 (automobile)  | 5 000 | 1 000 |
| dog      | CIFAR-10  | 5               | 5 000 | 1 000 |

Los datos se descargan automáticamente en `data/` la primera vez que se ejecuta el entrenamiento.

---

## Estructura del proyecto

```
├── config.py               # Hiperparámetros y rutas (fuente única de verdad)
├── train_pipeline.py       # Punto de entrada del entrenamiento
├── requirements.txt
│
├── src/
│   ├── dataset.py          # Filtrado y fusión de CIFAR-10 + CIFAR-100
│   ├── model.py            # ResNet18 + helpers de fine-tuning
│   ├── train.py            # Bucle de entrenamiento, evaluación y métricas
│   └── transforms.py       # Augmentación (train) y preprocesado (val/infer)
│
├── api/
│   ├── main.py             # FastAPI app, endpoints, CORS, static files
│   └── inference.py        # Singleton del modelo + predict_bytes()
│
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js              # Estado: idle → loading → result → idle
```

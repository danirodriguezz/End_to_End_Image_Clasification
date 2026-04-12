# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Train the model:**
```bash
python train_pipeline.py
```
Downloads CIFAR-10/100 automatically on first run, saves `models/best_model_weights.pth` and `models/model_metadata.json`.

**Run the API server:**
```bash
uvicorn api.main:app --reload --port 8000
```
- Web UI: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Architecture

This is an end-to-end image classification system for 4 classes: **airplane, bicycle, car, dog**.

### Data Pipeline
`src/dataset.py` merges two sources into one dataset:
- CIFAR-10 → airplane (class 0), car (class 1), dog (class 5)
- CIFAR-100 → bicycle (class 8)

Bicycle is ~10× underrepresented (~500 vs ~5000 samples). This is handled with `WeightedRandomSampler` (10× oversample) and `CrossEntropyLoss` with class weight `[1.0, 10.0, 1.0, 1.0]`.

### Two-Phase Fine-Tuning (`src/train.py`)
1. **Phase 1** (5 epochs): Backbone frozen, only FC head trained (lr=1e-3)
2. **Phase 2** (15 epochs): Full network, differential LRs — backbone 1e-4, head 1e-3

### API (`api/`)
FastAPI app in `api/main.py`. The model is loaded once at import time via `api/inference.py` — not per-request. `POST /predict` accepts multipart file upload and returns confidence scores sorted descending.

### Configuration
All hyperparameters, paths, class mappings, and device selection live in `config.py`. Nothing is hardcoded elsewhere.

### Frontend (`frontend/`)
Static files served by FastAPI. `app.js` implements an idle → loading → result → idle state machine. Uploads files via `POST /predict`, animates confidence bars on response.
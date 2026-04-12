"""
FastAPI backend — image classification API.

Endpoints
─────────
  POST /predict          Upload an image, get class probabilities
  GET  /health           Liveness check
  GET  /classes          List supported class names

Static frontend is served from /  (the frontend/ directory).

Start the server:
  uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import CLASS_NAMES, BASE_DIR

# Import here so the model loads at startup, not on the first request
from api.inference import predict_bytes


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Image Classifier API",
    description="Classifies images as airplane, bicycle, car, or dog.",
    version="1.0.0",
)

# CORS — allows the frontend (even if served from a different origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/classes", tags=["meta"])
def classes():
    """Return the list of classes the model can predict."""
    return {"classes": CLASS_NAMES}


@app.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(..., description="Image to classify (JPEG/PNG/WebP)")):
    """
    Upload an image and receive confidence scores for each class.

    Returns:
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
    """
    # Validate MIME type loosely (client can lie, PIL will fail safely if wrong)
    content_type = file.content_type or ""
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Please upload an image.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        result = predict_bytes(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the image: {exc}",
        )

    return JSONResponse(content=result)


# ── Serve frontend AFTER defining API routes (order matters in FastAPI) ────────

_frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

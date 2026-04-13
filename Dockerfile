FROM python:3.11-slim

WORKDIR /app

# Librerías de sistema requeridas por Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch CPU-only primero (evita descargar el build CUDA de ~2 GB) ────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
      "torch==2.2.2" "torchvision==0.17.2" \
      --index-url https://download.pytorch.org/whl/cpu

# ── Resto de dependencias (torch ya satisfecho, se omite) ────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Código fuente ─────────────────────────────────────────────────────────────
COPY . .

# HF Spaces requiere exponer el puerto 7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

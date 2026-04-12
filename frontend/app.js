/**
 * Image Classifier frontend
 * ─────────────────────────
 * States: idle → loading → result → idle
 *
 * The /predict API endpoint must be running at API_URL.
 * Because FastAPI serves this file as a static asset, relative paths work.
 */

const API_URL = "/predict";

// ── DOM refs ───────────────────────────────────────────────────────────────────
const dropZone             = document.getElementById("drop-zone");
const fileInput            = document.getElementById("file-input");
const uploadSection        = document.getElementById("upload-section");
const resultSection        = document.getElementById("result-section");
const previewImg           = document.getElementById("preview-img");
const loadingIndicator     = document.getElementById("loading-indicator");
const predictionsContainer = document.getElementById("predictions-container");
const topLabel             = document.getElementById("top-label");
const barsContainer        = document.getElementById("bars");
const resetBtn             = document.getElementById("reset-btn");


// ── State machine ──────────────────────────────────────────────────────────────

function showIdle() {
  uploadSection.hidden  = false;
  resultSection.hidden  = true;
  fileInput.value       = "";
  previewImg.src        = "";
}

function showLoading(objectUrl) {
  uploadSection.hidden         = false;   // keep upload section visible for layout
  resultSection.hidden         = false;
  previewImg.src               = objectUrl;
  loadingIndicator.hidden      = false;
  predictionsContainer.hidden  = true;
  uploadSection.hidden         = true;
}

function showResult(objectUrl, data) {
  resultSection.hidden         = false;
  previewImg.src               = objectUrl;
  loadingIndicator.hidden      = true;
  predictionsContainer.hidden  = false;

  // Top class label
  topLabel.textContent = `Es un ${data.top_class}`;

  // Build bar rows
  barsContainer.innerHTML = "";
  data.predictions.forEach((item, idx) => {
    const pct = (item.confidence * 100).toFixed(1);
    const isTop = idx === 0;

    const row = document.createElement("div");
    row.className = "bar-row";
    row.dataset.class = item.class;

    row.innerHTML = `
      <div class="bar-header">
        <span class="bar-name">${item.class}</span>
        <span class="bar-pct">${pct}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${isTop ? "top" : ""}" style="width: 0%"></div>
      </div>
    `;
    barsContainer.appendChild(row);

    // Animate after a tick so the CSS transition fires
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        row.querySelector(".bar-fill").style.width = `${pct}%`;
      });
    });
  });
}


// ── Upload & predict ───────────────────────────────────────────────────────────

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Por favor selecciona un archivo de imagen (JPEG, PNG, WebP).");
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  showLoading(objectUrl);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(API_URL, { method: "POST", body: formData });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Error del servidor: ${response.status}`);
    }

    const data = await response.json();
    showResult(objectUrl, data);

  } catch (error) {
    alert(`No se pudo clasificar la imagen.\n${error.message}`);
    showIdle();
    URL.revokeObjectURL(objectUrl);
  }
}


// ── Event listeners ────────────────────────────────────────────────────────────

// Click to open file browser
dropZone.addEventListener("click", () => fileInput.click());

// Keyboard accessibility
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

// File selected via browser dialog
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

// Drag & drop
dropZone.addEventListener("dragover",  (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// Reset button
resetBtn.addEventListener("click", showIdle);

// Initial state
showIdle();

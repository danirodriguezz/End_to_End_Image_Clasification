/* viz.js — Three.js 3D visualizations */

// ── Shared resize helper ───────────────────────────────────────────────────────
function syncRenderer(renderer, camera, canvas) {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight || 1;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

// ── 1. Hero — animated neural network ─────────────────────────────────────────
function initHero() {
  const canvas = document.getElementById('hero-canvas');
  if (!canvas || !window.THREE) return;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, 2, 0.1, 200);
  camera.position.set(0, 0, 18);

  // Layer node counts — like a wide MLP
  const layerCounts = [3, 6, 9, 12, 9, 6, 4];
  const layerSpX = 3.8;
  const nodeColor  = new THREE.Color(0x6366f1);
  const pulseColor = new THREE.Color(0x06b6d4);

  const nodeGeo = new THREE.SphereGeometry(0.13, 16, 16);
  const nodeMat = new THREE.MeshBasicMaterial({ color: nodeColor });

  // Build layers
  const layers = layerCounts.map((count, li) => {
    const x = (li - (layerCounts.length - 1) / 2) * layerSpX;
    return Array.from({ length: count }, (_, ni) => {
      const mesh = new THREE.Mesh(nodeGeo, nodeMat.clone());
      mesh.position.set(x, (ni - (count - 1) / 2) * 1.3, (Math.random() - 0.5) * 0.6);
      scene.add(mesh);
      return mesh;
    });
  });

  // Connections (sparse)
  const lineMat = new THREE.LineBasicMaterial({ color: 0x4338ca, transparent: true, opacity: 0.13 });
  layers.forEach((layer, li) => {
    if (li >= layers.length - 1) return;
    layer.forEach(a => {
      layers[li + 1].forEach(b => {
        if (Math.random() > 0.35) {
          const geo = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
          scene.add(new THREE.Line(geo, lineMat.clone()));
        }
      });
    });
  });

  // Pulses
  const pulseGeo = new THREE.SphereGeometry(0.07, 8, 8);
  const activePulses = [];

  function spawnPulse() {
    const li = Math.floor(Math.random() * (layers.length - 1));
    const a  = layers[li][Math.floor(Math.random() * layers[li].length)];
    const b  = layers[li + 1][Math.floor(Math.random() * layers[li + 1].length)];
    const mesh = new THREE.Mesh(pulseGeo, new THREE.MeshBasicMaterial({ color: pulseColor, transparent: true }));
    mesh.position.copy(a.position);
    scene.add(mesh);
    activePulses.push({ mesh, from: a.position.clone(), to: b.position.clone(), t: 0, spd: 0.5 + Math.random() * 0.5 });
  }
  const pulseInterval = setInterval(spawnPulse, 140);

  // Animate
  let t = 0;
  function animate() {
    requestAnimationFrame(animate);
    t += 0.004;

    camera.position.x = Math.sin(t * 0.7) * 3;
    camera.position.y = Math.sin(t * 0.4) * 2;
    camera.lookAt(0, 0, 0);

    // Node breathe
    layers.flat().forEach((n, i) => {
      n.scale.setScalar(0.8 + 0.25 * Math.sin(t * 1.8 + i * 0.6));
    });

    // Pulse travel
    for (let i = activePulses.length - 1; i >= 0; i--) {
      const p = activePulses[i];
      p.t += p.spd * 0.016;
      if (p.t >= 1) {
        scene.remove(p.mesh);
        activePulses.splice(i, 1);
      } else {
        p.mesh.position.lerpVectors(p.from, p.to, p.t);
        p.mesh.material.opacity = Math.sin(p.t * Math.PI) * 0.9;
      }
    }

    syncRenderer(renderer, camera, canvas);
    renderer.render(scene, camera);
  }
  animate();

  // Cleanup on page unload
  window.addEventListener('beforeunload', () => clearInterval(pulseInterval));
}

// ── 2. Architecture — ResNet18 layer blocks ────────────────────────────────────
function initArchitecture() {
  const canvas = document.getElementById('arch-canvas');
  if (!canvas || !window.THREE) return;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 1.5, 13);
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const sun = new THREE.DirectionalLight(0xffffff, 0.9);
  sun.position.set(4, 8, 6);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x6366f1, 0.3);
  fill.position.set(-4, -2, -4);
  scene.add(fill);

  // [label, channels, spatial_px, color]
  const layerDefs = [
    { label: 'Input\n224²×3',    ch: 3,   sp: 224, color: 0x475569 },
    { label: 'Conv1+Pool\n56²×64',  ch: 64,  sp: 56,  color: 0x6366f1 },
    { label: 'Layer 1\n56²×64',  ch: 64,  sp: 56,  color: 0x7c3aed },
    { label: 'Layer 2\n28²×128', ch: 128, sp: 28,  color: 0x0891b2 },
    { label: 'Layer 3\n14²×256', ch: 256, sp: 14,  color: 0x0284c7 },
    { label: 'Layer 4\n7²×512',  ch: 512, sp: 7,   color: 0x059669 },
    { label: 'FC → 4',           ch: 4,   sp: 1,   color: 0xd97706 },
  ];

  const group = new THREE.Group();
  scene.add(group);

  const totalW = (layerDefs.length - 1) * 1.8;

  layerDefs.forEach((ld, i) => {
    const H = Math.log2(ld.ch + 1) / Math.log2(513) * 5 + 0.25;
    const D = Math.log2(ld.sp + 1) / Math.log2(225) * 3.5 + 0.15;
    const W = 0.65;
    const x = i * 1.8 - totalW / 2;

    const geo  = new THREE.BoxGeometry(W, H, D);
    const mat  = new THREE.MeshPhongMaterial({ color: ld.color, shininess: 90, specular: 0x222222 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, H / 2, 0);
    group.add(mesh);

    // Thin connector line to next block
    if (i < layerDefs.length - 1) {
      const nextH = Math.log2(layerDefs[i+1].ch + 1) / Math.log2(513) * 5 + 0.25;
      const pts = [
        new THREE.Vector3(x + W / 2,      H / 2,      0),
        new THREE.Vector3(x + 1.8 - W / 2, nextH / 2, 0),
      ];
      const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
      group.add(new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.6 })));
    }
  });

  // Drag to rotate
  let drag = false, prevX = 0, prevY = 0, rotY = -0.25, rotX = 0.18;
  const onDown = (x, y) => { drag = true; prevX = x; prevY = y; };
  const onMove = (x, y) => {
    if (!drag) return;
    rotY += (x - prevX) * 0.012;
    rotX  = Math.max(-0.55, Math.min(0.55, rotX + (y - prevY) * 0.006));
    prevX = x; prevY = y;
  };
  const onUp = () => { drag = false; };

  canvas.addEventListener('mousedown',  e => onDown(e.clientX, e.clientY));
  canvas.addEventListener('mousemove',  e => onMove(e.clientX, e.clientY));
  canvas.addEventListener('mouseup',    onUp);
  canvas.addEventListener('mouseleave', onUp);
  canvas.addEventListener('touchstart', e => onDown(e.touches[0].clientX, e.touches[0].clientY), { passive: true });
  canvas.addEventListener('touchmove',  e => onMove(e.touches[0].clientX, e.touches[0].clientY), { passive: true });
  canvas.addEventListener('touchend',   onUp);

  function animate() {
    requestAnimationFrame(animate);
    if (!drag) rotY += 0.004;
    group.rotation.y = rotY;
    group.rotation.x = rotX;
    syncRenderer(renderer, camera, canvas);
    renderer.render(scene, camera);
  }
  animate();
}

// ── 3. Dataset — 3D bar chart ──────────────────────────────────────────────────
function initDatasetChart() {
  const canvas = document.getElementById('chart-canvas');
  if (!canvas || !window.THREE) return;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 2.5, 12);
  camera.lookAt(0, 1.5, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const sun = new THREE.DirectionalLight(0xffffff, 1.0);
  sun.position.set(3, 8, 5);
  scene.add(sun);
  const back = new THREE.DirectionalLight(0x3b82f6, 0.25);
  back.position.set(-3, -1, -5);
  scene.add(back);

  const chartData = [
    { label: 'Avión',     count: 5000, color: 0x6366f1 },
    { label: 'Bicicleta', count: 500,  color: 0xf59e0b },
    { label: 'Coche',     count: 5000, color: 0x06b6d4 },
    { label: 'Perro',     count: 5000, color: 0x10b981 },
  ];
  const maxH = 5;
  const spacing = 1.8;
  const totalW = (chartData.length - 1) * spacing;

  const bars = chartData.map((d, i) => {
    const targetH = (d.count / 5000) * maxH;
    const x = i * spacing - totalW / 2;
    const geo  = new THREE.BoxGeometry(1.1, targetH, 1.1);
    const mat  = new THREE.MeshPhongMaterial({ color: d.color, shininess: 80, specular: 0x222222 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, targetH / 2, 0);
    mesh.scale.y = 0.001;
    scene.add(mesh);

    // Base plate
    const plateGeo = new THREE.BoxGeometry(1.15, 0.06, 1.15);
    const plateMat = new THREE.MeshPhongMaterial({ color: new THREE.Color(d.color).multiplyScalar(0.5) });
    const plate = new THREE.Mesh(plateGeo, plateMat);
    plate.position.set(x, 0.03, 0);
    scene.add(plate);

    return { mesh, targetH };
  });

  // Ground grid
  const grid = new THREE.GridHelper(12, 12, 0x1e293b, 0x1e293b);
  grid.position.y = 0;
  scene.add(grid);

  // Scroll-triggered growth
  let grown = false;
  let progress = 0;
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) grown = true;
  }, { threshold: 0.3 });
  obs.observe(canvas);

  let camT = 0;
  function animate() {
    requestAnimationFrame(animate);
    camT += 0.004;

    if (grown && progress < 1) {
      progress = Math.min(progress + 0.018, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      bars.forEach(b => {
        b.mesh.scale.y = ease;
        b.mesh.position.y = b.targetH * ease / 2;
      });
    }

    camera.position.x = Math.sin(camT * 0.25) * 2.5;
    camera.position.z = 12 + Math.cos(camT * 0.25) * 1.5;
    camera.lookAt(0, 1.5, 0);

    syncRenderer(renderer, camera, canvas);
    renderer.render(scene, camera);
  }
  animate();
}

// ── Navbar scroll effect & smooth links ────────────────────────────────────────
function initNav() {
  const nav = document.getElementById('main-nav');
  if (!nav) return;

  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 60);
  }, { passive: true });

  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

// ── Scroll-reveal ──────────────────────────────────────────────────────────────
function initReveal() {
  const els = document.querySelectorAll('.reveal');
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('revealed'); obs.unobserve(e.target); }
    });
  }, { threshold: 0.12 });
  els.forEach(el => obs.observe(el));
}

// ── Weight bars animate ────────────────────────────────────────────────────────
function initWeightBars() {
  const bars = document.querySelectorAll('.weight-fill');
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) {
      bars.forEach(b => { b.style.width = b.dataset.pct; });
      obs.disconnect();
    }
  }, { threshold: 0.4 });
  if (bars.length) obs.observe(bars[0].closest('section') || bars[0]);
}

// ── Boot ───────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initNav();
  initReveal();
  initWeightBars();
  initHero();
  initArchitecture();
  initDatasetChart();
});

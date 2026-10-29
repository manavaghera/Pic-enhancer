// Before/after compare viewer with a draggable split, wheel/pinch zoom and
// panning. Both images are laid out in output-pixel units and share one
// transform, so the original is shown exactly as a plain browser upscale.

import { h } from './dom.js';

const MAX_ZOOM = 16;
const ARROWS = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 7-5 5 5 5M15 7l5 5-5 5"/></svg>';

export function createViewer(root, { onZoomChange } = {}) {
  const beforeImg = h('img', { class: 'vimg', alt: 'Original image', draggable: 'false' });
  const afterImg = h('img', { class: 'vimg', alt: 'Enhanced image', draggable: 'false' });
  const afterLayer = h('div', { class: 'vlayer vlayer-after' }, afterImg);
  const handle = h('div', {
    class: 'vhandle',
    role: 'slider',
    tabindex: 0,
    'aria-label': 'Before and after split position',
    'aria-valuemin': 0,
    'aria-valuemax': 100,
    html: ARROWS,
  });
  const divider = h('div', { class: 'vdivider' }, handle);
  root.append(
    h('div', { class: 'vlayer vlayer-before' }, beforeImg),
    afterLayer,
    divider,
    h('span', { class: 'vtag vtag-before' }, 'Before'),
    h('span', { class: 'vtag vtag-after' }, 'After'),
  );

  const s = { w: 0, h: 0, zoom: 1, fit: 1, x: 0, y: 0, split: 0.5, compare: false, fitted: true };

  const size = () => ({ vw: root.clientWidth, vh: root.clientHeight });

  function apply() {
    const t = `translate(${s.x}px, ${s.y}px) scale(${s.zoom})`;
    beforeImg.style.transform = t;
    afterImg.style.transform = t;
    const px = Math.round(s.split * size().vw);
    afterLayer.style.clipPath = `inset(0 0 0 ${px}px)`;
    divider.style.transform = `translateX(${px}px)`;
    handle.setAttribute('aria-valuenow', String(Math.round(s.split * 100)));
    root.classList.toggle('is-compare', s.compare);
    root.classList.toggle('is-zoomed', s.zoom > s.fit * 1.001);
    onZoomChange?.(s.zoom);
  }

  function clampPan() {
    const { vw, vh } = size();
    const iw = s.w * s.zoom;
    const ih = s.h * s.zoom;
    s.x = iw <= vw ? (vw - iw) / 2 : Math.min(0, Math.max(vw - iw, s.x));
    s.y = ih <= vh ? (vh - ih) / 2 : Math.min(0, Math.max(vh - ih, s.y));
  }

  function fit() {
    const { vw, vh } = size();
    if (!s.w || !vw || !vh) return;
    s.fit = Math.min(Math.max(1, vw - 32) / s.w, Math.max(1, vh - 32) / s.h, 8);
    s.zoom = s.fit;
    s.fitted = true;
    clampPan();
    apply();
  }

  function zoomTo(z, cx, cy) {
    if (!s.w) return;
    const { vw, vh } = size();
    const ox = cx ?? vw / 2;
    const oy = cy ?? vh / 2;
    const nz = Math.min(MAX_ZOOM, Math.max(s.fit * 0.5, z));
    s.x = ox - (ox - s.x) * (nz / s.zoom);
    s.y = oy - (oy - s.y) * (nz / s.zoom);
    s.zoom = nz;
    s.fitted = Math.abs(nz - s.fit) < 1e-6;
    clampPan();
    apply();
  }

  const local = (e) => {
    const r = root.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top, width: r.width };
  };

  function setSplit(e) {
    const p = local(e);
    s.split = Math.min(1, Math.max(0, p.x / Math.max(1, p.width)));
    apply();
  }

  const pointers = new Map();
  let mode = null;
  let last = null;
  let pinch = null;

  const pinchInfo = () => {
    const [a, b] = [...pointers.values()];
    return { dist: Math.hypot(a.x - b.x, a.y - b.y) || 1, mid: { clientX: (a.x + b.x) / 2, clientY: (a.y + b.y) / 2 } };
  };

  root.addEventListener('pointerdown', (e) => {
    if (!s.w || (e.pointerType === 'mouse' && e.button !== 0)) return;
    root.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      mode = 'pinch';
      pinch = { ...pinchInfo(), zoom: s.zoom };
      return;
    }
    const zoomed = s.zoom > s.fit * 1.001;
    mode = s.compare && (divider.contains(e.target) || !zoomed) ? 'split' : 'pan';
    if (mode === 'split') setSplit(e);
    last = { x: e.clientX, y: e.clientY };
  });

  root.addEventListener('pointermove', (e) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (mode === 'pinch' && pointers.size === 2) {
      const now = pinchInfo();
      const p = local(now.mid);
      zoomTo(pinch.zoom * (now.dist / pinch.dist), p.x, p.y);
    } else if (mode === 'split') {
      setSplit(e);
    } else if (mode === 'pan' && last) {
      s.x += e.clientX - last.x;
      s.y += e.clientY - last.y;
      last = { x: e.clientX, y: e.clientY };
      s.fitted = false;
      clampPan();
      apply();
    }
  });

  const release = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size === 1 && mode === 'pinch') {
      const [p] = pointers.values();
      mode = 'pan';
      last = { x: p.x, y: p.y };
    } else if (pointers.size === 0) {
      mode = null;
      last = null;
    }
  };
  root.addEventListener('pointerup', release);
  root.addEventListener('pointercancel', release);

  root.addEventListener(
    'wheel',
    (e) => {
      if (!s.w) return;
      e.preventDefault();
      const p = local(e);
      const k = e.deltaMode === 1 ? 0.05 : 0.0015;
      zoomTo(s.zoom * Math.exp(-e.deltaY * k), p.x, p.y);
    },
    { passive: false },
  );

  root.addEventListener('dblclick', (e) => {
    if (!s.w) return;
    const p = local(e);
    if (!s.fitted) fit();
    else zoomTo(s.fit >= 1 ? s.fit * 2 : 1, p.x, p.y);
  });

  handle.addEventListener('keydown', (e) => {
    const steps = { ArrowLeft: -0.02, ArrowRight: 0.02, ArrowDown: -0.02, ArrowUp: 0.02 };
    if (e.key in steps) s.split = Math.min(1, Math.max(0, s.split + steps[e.key]));
    else if (e.key === 'Home') s.split = 0;
    else if (e.key === 'End') s.split = 1;
    else return;
    e.preventDefault();
    apply();
  });

  new ResizeObserver(() => {
    if (!s.w) return;
    if (s.fitted) fit();
    else {
      clampPan();
      apply();
    }
  }).observe(root);

  return {
    // width/height are the display (output) pixel size; reset re-fits the view.
    show({ beforeUrl, afterUrl = null, width, height, reset = false }) {
      const resized = width !== s.w || height !== s.h;
      s.w = width;
      s.h = height;
      for (const img of [beforeImg, afterImg]) {
        img.style.width = `${width}px`;
        img.style.height = `${height}px`;
      }
      if (beforeImg.getAttribute('src') !== beforeUrl) beforeImg.src = beforeUrl;
      if (afterUrl) {
        if (afterImg.getAttribute('src') !== afterUrl) afterImg.src = afterUrl;
      } else {
        afterImg.removeAttribute('src');
      }
      s.compare = Boolean(afterUrl);
      if (resized || reset) fit();
      else apply();
    },
    clear() {
      s.w = 0;
      s.h = 0;
      beforeImg.removeAttribute('src');
      afterImg.removeAttribute('src');
      s.compare = false;
      root.classList.remove('is-compare', 'is-zoomed');
    },
    fit,
    zoomIn: () => zoomTo(s.zoom * 1.25),
    zoomOut: () => zoomTo(s.zoom / 1.25),
    actualSize: () => zoomTo(1),
  };
}

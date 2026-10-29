// Separable Lanczos-3 resampling (with an anti-aliased kernel when shrinking)
// and nearest-neighbour for pixel art. Transparent images are resampled in
// premultiplied alpha so edges do not pick up dark or coloured fringes.

import { cloneImage, createImage, hasTransparency, premultiply, unpremultiply } from './image.js';

const LOBES = 3;

function lanczos(x) {
  if (x === 0) return 1;
  if (x <= -LOBES || x >= LOBES) return 0;
  const px = Math.PI * x;
  return (LOBES * Math.sin(px) * Math.sin(px / LOBES)) / (px * px);
}

export function buildWeights(srcSize, dstSize) {
  const scale = dstSize / srcSize;
  const fscale = Math.max(1, 1 / scale);
  const support = LOBES * fscale;
  const maxTaps = Math.ceil(support * 2) + 2;
  const starts = new Int32Array(dstSize);
  const counts = new Int32Array(dstSize);
  const weights = new Float32Array(dstSize * maxTaps);
  for (let i = 0; i < dstSize; i++) {
    const center = (i + 0.5) / scale - 0.5;
    let start = Math.max(0, Math.ceil(center - support));
    const end = Math.min(srcSize - 1, Math.floor(center + support));
    const off = i * maxTaps;
    let sum = 0;
    let count = 0;
    for (let j = start; j <= end; j++) {
      const wgt = lanczos((j - center) / fscale);
      weights[off + count] = wgt;
      sum += wgt;
      count++;
    }
    if (count === 0 || !(sum > 1e-8)) {
      start = Math.min(srcSize - 1, Math.max(0, Math.round(center)));
      weights[off] = 1;
      count = 1;
    } else {
      for (let k = 0; k < count; k++) weights[off + k] /= sum;
    }
    starts[i] = start;
    counts[i] = count;
  }
  return { starts, counts, weights, maxTaps };
}

function resampleRows(src, sw, sh, dw, onRow) {
  const { starts, counts, weights, maxTaps } = buildWeights(sw, dw);
  const out = new Uint8ClampedArray(dw * sh * 4);
  for (let y = 0; y < sh; y++) {
    const srow = y * sw * 4;
    const orow = y * dw * 4;
    for (let x = 0; x < dw; x++) {
      const off = x * maxTaps;
      const c = counts[x];
      let sp = srow + starts[x] * 4;
      let r = 0, g = 0, b = 0, a = 0;
      for (let k = 0; k < c; k++, sp += 4) {
        const wgt = weights[off + k];
        r += wgt * src[sp];
        g += wgt * src[sp + 1];
        b += wgt * src[sp + 2];
        a += wgt * src[sp + 3];
      }
      const o = orow + x * 4;
      out[o] = r;
      out[o + 1] = g;
      out[o + 2] = b;
      out[o + 3] = a;
    }
    if (onRow) onRow(y, sh);
  }
  return out;
}

function resampleColumns(src, w, sh, dh, onRow) {
  const { starts, counts, weights, maxTaps } = buildWeights(sh, dh);
  const stride = w * 4;
  const out = new Uint8ClampedArray(stride * dh);
  const acc = new Float32Array(stride);
  for (let y = 0; y < dh; y++) {
    acc.fill(0);
    const off = y * maxTaps;
    const c = counts[y];
    for (let k = 0; k < c; k++) {
      const wgt = weights[off + k];
      const srow = (starts[y] + k) * stride;
      for (let j = 0; j < stride; j++) acc[j] += wgt * src[srow + j];
    }
    out.set(acc, y * stride);
    if (onRow) onRow(y, dh);
  }
  return out;
}

function resizeNearest(img, dw, dh) {
  const { width: sw, height: sh, data: s } = img;
  const out = createImage(dw, dh);
  const d = out.data;
  const xmap = new Int32Array(dw);
  for (let x = 0; x < dw; x++) xmap[x] = Math.min(sw - 1, Math.floor(((x + 0.5) * sw) / dw)) * 4;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.floor(((y + 0.5) * sh) / dh));
    const srow = sy * sw * 4;
    let o = y * dw * 4;
    for (let x = 0; x < dw; x++, o += 4) {
      const sp = srow + xmap[x];
      d[o] = s[sp];
      d[o + 1] = s[sp + 1];
      d[o + 2] = s[sp + 2];
      d[o + 3] = s[sp + 3];
    }
  }
  return out;
}

// Returns a new image of dw x dh. method: 'lanczos' (default) or 'nearest'.
export function resize(img, dw, dh, { method = 'lanczos', onProgress } = {}) {
  dw = Math.max(1, Math.round(dw));
  dh = Math.max(1, Math.round(dh));
  if (dw === img.width && dh === img.height) return cloneImage(img);
  if (method === 'nearest') return resizeNearest(img, dw, dh);

  const alpha = hasTransparency(img);
  const work = alpha ? cloneImage(img) : img;
  if (alpha) premultiply(work);

  const { width: sw, height: sh } = work;
  const hShare = dw === sw ? 0 : (dw * sh) / (dw * sh + dw * dh);
  const report = (base, share) => (y, total) => {
    if (onProgress && (y & 63) === 0) onProgress(base + share * (y / total));
  };

  const rows = dw === sw ? work.data : resampleRows(work.data, sw, sh, dw, report(0, hShare));
  const data = dh === sh ? new Uint8ClampedArray(rows) : resampleColumns(rows, dw, sh, dh, report(hShare, 1 - hShare));
  const out = { width: dw, height: dh, data };
  if (alpha) unpremultiply(out);
  if (onProgress) onProgress(1);
  return out;
}

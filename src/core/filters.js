// Spatial filters: box/gaussian blur, guided-filter denoise, unsharp-mask
// sharpening and local-contrast "clarity". All heavy work runs per band
// (see mapBands) so memory stays bounded on 8K outputs.

import { cloneImage, mapBands, toLuma } from './image.js';

const clampIndex = (v, max) => (v < 0 ? 0 : v > max ? max : v);

// Box blur with clamp-to-edge borders. `out` may alias `src`; `tmp` must not.
export function boxBlur(src, w, h, r, out = new Float32Array(w * h), tmp = new Float32Array(w * h)) {
  if (r < 1) {
    if (out !== src) out.set(src);
    return out;
  }
  const norm = 1 / (2 * r + 1);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += src[row + clampIndex(k, w - 1)];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = acc * norm;
      acc += src[row + clampIndex(x + r + 1, w - 1)] - src[row + clampIndex(x - r, w - 1)];
    }
  }
  const acc = new Float64Array(w);
  for (let k = -r; k <= r; k++) {
    const row = clampIndex(k, h - 1) * w;
    for (let x = 0; x < w; x++) acc[x] += tmp[row + x];
  }
  for (let y = 0; y < h; y++) {
    const o = y * w;
    const add = clampIndex(y + r + 1, h - 1) * w;
    const sub = clampIndex(y - r, h - 1) * w;
    for (let x = 0; x < w; x++) {
      out[o + x] = acc[x] * norm;
      acc[x] += tmp[add + x] - tmp[sub + x];
    }
  }
  return out;
}

const boxRadiusFor = (sigma) => Math.max(1, Math.round((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2));

// Pixels of context a gaussianBlur(sigma) reads on each side.
export function gaussianSupport(sigma) {
  return sigma < 3 ? Math.max(1, Math.ceil(sigma * 3)) : 3 * boxRadiusFor(sigma);
}

// Exact separable kernel for small sigma, three box passes for large sigma.
export function gaussianBlur(src, w, h, sigma) {
  if (sigma <= 0) return Float32Array.from(src);
  if (sigma >= 3) {
    const r = boxRadiusFor(sigma);
    const tmp = new Float32Array(w * h);
    const a = boxBlur(src, w, h, r, new Float32Array(w * h), tmp);
    boxBlur(a, w, h, r, a, tmp);
    return boxBlur(a, w, h, r, a, tmp);
  }
  const rad = Math.max(1, Math.ceil(sigma * 3));
  const klen = 2 * rad + 1;
  const k = new Float32Array(klen);
  let sum = 0;
  for (let i = 0; i < klen; i++) {
    const t = i - rad;
    k[i] = Math.exp(-(t * t) / (2 * sigma * sigma));
    sum += k[i];
  }
  for (let i = 0; i < klen; i++) k[i] /= sum;

  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      if (x >= rad && x < w - rad) {
        const base = row + x - rad;
        for (let i = 0; i < klen; i++) acc += k[i] * src[base + i];
      } else {
        for (let i = 0; i < klen; i++) acc += k[i] * src[row + clampIndex(x + i - rad, w - 1)];
      }
      tmp[row + x] = acc;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let i = 0; i < klen; i++) {
      const rr = clampIndex(y + i - rad, h - 1) * w;
      const kk = k[i];
      for (let x = 0; x < w; x++) out[o + x] += kk * tmp[rr + x];
    }
  }
  return out;
}

// Guided filter (He et al. 2010). I is the guide, p the input; support is 2r.
export function guidedFilter(I, p, w, h, r, eps) {
  const n = w * h;
  const tmp = new Float32Array(n);
  const meanI = boxBlur(I, w, h, r, new Float32Array(n), tmp);
  const meanP = I === p ? meanI : boxBlur(p, w, h, r, new Float32Array(n), tmp);
  const ip = new Float32Array(n);
  for (let i = 0; i < n; i++) ip[i] = I[i] * p[i];
  const corrIP = boxBlur(ip, w, h, r, ip, tmp);
  let corrII = corrIP;
  if (I !== p) {
    const ii = new Float32Array(n);
    for (let i = 0; i < n; i++) ii[i] = I[i] * I[i];
    corrII = boxBlur(ii, w, h, r, ii, tmp);
  }
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const mI = meanI[i];
    const mP = meanP[i];
    const cov = corrIP[i] - mI * mP;
    const variance = corrII[i] - mI * mI;
    const ai = cov / (variance + eps);
    a[i] = ai;
    b[i] = mP - ai * mI;
  }
  boxBlur(a, w, h, r, a, tmp);
  boxBlur(b, w, h, r, b, tmp);
  const q = new Float32Array(n);
  for (let i = 0; i < n; i++) q[i] = a[i] * I[i] + b[i];
  return q;
}

function denoiseBand(band, ry, rc, epsY, epsC, keep) {
  const { width: w, height: h, data: d } = band;
  const n = w * h;
  const Y = new Float32Array(n);
  const Cb = new Float32Array(n);
  const Cr = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = d[p], g = d[p + 1], b = d[p + 2];
    Y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    Cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b;
    Cr[i] = 0.5 * r - 0.418688 * g - 0.081312 * b;
  }
  const Yq = guidedFilter(Y, Y, w, h, ry, epsY);
  const Cbq = guidedFilter(Y, Cb, w, h, rc, epsC);
  const Crq = guidedFilter(Y, Cr, w, h, rc, epsC);
  const out = new Uint8ClampedArray(d.length);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const y = Yq[i] + (Y[i] - Yq[i]) * keep;
    const cb = Cbq[i];
    const cr = Crq[i];
    out[p] = y + 1.402 * cr;
    out[p + 1] = y - 0.344136 * cb - 0.714136 * cr;
    out[p + 2] = y + 1.772 * cb;
    out[p + 3] = d[p + 3];
  }
  return out;
}

// Edge-preserving noise reduction in YCbCr; chroma is smoothed harder than
// luma because colour noise is the most visible. strength is 0..1.
export function denoise(img, strength, onProgress) {
  if (!(strength > 0)) return cloneImage(img);
  const s = Math.min(1, strength);
  const ry = 2 + Math.round(2 * s);
  const rc = ry + 2;
  const epsY = (3 + 22 * s) ** 2;
  const epsC = (6 + 34 * s) ** 2;
  const keep = 0.3 * (1 - s);
  return mapBands(img, 2 * rc + 2, (band) => denoiseBand(band, ry, rc, epsY, epsC, keep), onProgress);
}

// Unsharp mask on luminance only, so edges get crisper without colour fringes.
// threshold (0..255) suppresses gain on low-amplitude detail such as noise;
// protectTones weights the effect towards midtones (used for clarity).
export function sharpen(img, { amount, sigma = 1, threshold = 2, maxDelta = 48, protectTones = false, onProgress }) {
  if (!(amount > 0)) return cloneImage(img);
  return mapBands(img, gaussianSupport(sigma) + 2, (band) => {
    const { width: w, height: h, data: src } = band;
    const Y = toLuma(band);
    const B = gaussianBlur(Y, w, h, sigma);
    const out = new Uint8ClampedArray(src);
    for (let i = 0, p = 0; i < Y.length; i++, p += 4) {
      const diff = Y[i] - B[i];
      const ad = diff < 0 ? -diff : diff;
      const k = threshold > 0 ? Math.min(1, (ad - threshold) / threshold) : 1;
      if (k <= 0) continue;
      let delta = diff * amount * k;
      if (protectTones) {
        const t = (2 * Y[i]) / 255 - 1;
        delta *= 1 - t * t;
      }
      if (delta > maxDelta) delta = maxDelta;
      else if (delta < -maxDelta) delta = -maxDelta;
      out[p] = src[p] + delta;
      out[p + 1] = src[p + 1] + delta;
      out[p + 2] = src[p + 2] + delta;
    }
    return out;
  }, onProgress);
}

// Large-radius, midtone-weighted local contrast boost. strength is 0..1.
export function clarity(img, strength, onProgress) {
  if (!(strength > 0)) return cloneImage(img);
  const sigma = Math.max(4, Math.min(40, Math.max(img.width, img.height) / 60));
  return sharpen(img, { amount: 0.9 * strength, sigma, threshold: 0, maxDelta: 36, protectTones: true, onProgress });
}

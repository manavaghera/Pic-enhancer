// Image quality analysis: sharpness, noise, exposure, contrast, colour,
// compression artifacts, resolution and an overall 0-100 score.

import { hasTransparency } from './image.js';
import { gaussianBlur } from './filters.js';
import { resize } from './resize.js';
import { dominantColors } from './palette.js';
import { neutralMean, percentile } from './color.js';

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const score = (v) => Math.round(100 * clamp01(v));

// Luma of the image composited over white, so transparent areas read as paper.
function opaqueLuma(img) {
  const d = img.data;
  const n = img.width * img.height;
  const L = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const a = d[p + 3] / 255;
    const y = 0.2126 * d[p] + 0.7152 * d[p + 1] + 0.0722 * d[p + 2];
    L[i] = y * a + 255 * (1 - a);
  }
  return L;
}

export function laplacianVariance(L, w, h) {
  if (w < 3 || h < 3) return 0;
  let sum = 0, sumSq = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const v = L[i - 1] + L[i + 1] + L[i - w] + L[i + w] - 4 * L[i];
      sum += v;
      sumSq += v * v;
      n++;
    }
  }
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

// Mean edge width in pixels on the strongest edges, from |Laplacian| / |gradient|
// (a step spread over s pixels gives ~s). Pre-smoothing with sigma 1 keeps
// noise from reading as detail. Contrast-independent, unlike Laplacian
// variance. Returns null when there are too few real edges to judge.
export function edgeWidth(L, w, h) {
  if (w < 5 || h < 5) return null;
  const S = gaussianBlur(L, w, h, 1);
  const iw = w - 2, ih = h - 2, n = iw * ih;
  const G = new Float32Array(n);
  const hist = new Uint32Array(1024);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const g = Math.hypot((S[i + 1] - S[i - 1]) / 2, (S[i + w] - S[i - w]) / 2);
      G[(y - 1) * iw + x - 1] = g;
      hist[Math.min(1023, Math.round(g * 4))]++;
    }
  }
  const thr = Math.max(3, percentile(hist, n, 0.9) / 4);
  let lap = 0, grad = 0, count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const k = (y - 1) * iw + x - 1;
      if (G[k] < thr) continue;
      const i = y * w + x;
      lap += Math.abs(S[i - 1] + S[i + 1] + S[i - w] + S[i + w] - 4 * S[i]);
      grad += G[k];
      count++;
    }
  }
  if (count < Math.max(16, n * 0.002)) return null;
  return lap > 0 ? Math.min(50, (2 * grad) / lap) : 50;
}

// 2x2 box-averaged luma; grain that JPEG smoothed at full scale shows here.
function halfLuma(L, w, h) {
  const hw = w >> 1, hh = h >> 1;
  const out = new Float32Array(hw * hh);
  for (let y = 0; y < hh; y++) {
    for (let x = 0; x < hw; x++) {
      const i = 2 * y * w + 2 * x;
      out[y * hw + x] = (L[i] + L[i + 1] + L[i + w] + L[i + w + 1]) / 4;
    }
  }
  return { L: out, w: hw, h: hh };
}

// Visible grain: the larger of the full- and half-scale estimates. White noise
// halves at half scale, so it is measured at full scale; compressed, blotchy
// grain survives downsampling and is caught at half scale.
export function estimateGrain(L, w, h) {
  const half = halfLuma(L, w, h);
  return Math.max(estimateNoise(L, w, h), estimateNoise(half.L, half.w, half.h));
}

// Immerkaer (1996) fast noise estimate, skipping the 10% strongest Sobel
// edges (Tai & Yang 2008) so image structure is not mistaken for noise.
export function estimateNoise(L, w, h) {
  if (w < 3 || h < 3) return 0;
  const iw = w - 2, ih = h - 2;
  const G = new Uint16Array(iw * ih);
  const hist = new Uint32Array(2041);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = L[i - w + 1] + 2 * L[i + 1] + L[i + w + 1] - L[i - w - 1] - 2 * L[i - 1] - L[i + w - 1];
      const gy = L[i + w - 1] + 2 * L[i + w] + L[i + w + 1] - L[i - w - 1] - 2 * L[i - w] - L[i - w + 1];
      const g = Math.min(2040, Math.round(Math.abs(gx) + Math.abs(gy)));
      G[(y - 1) * iw + (x - 1)] = g;
      hist[g]++;
    }
  }
  const thr = percentile(hist, iw * ih, 0.9);
  let sum = 0, count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      if (G[(y - 1) * iw + (x - 1)] > thr) continue;
      const i = y * w + x;
      const v =
        L[i - w - 1] - 2 * L[i - w] + L[i - w + 1] -
        2 * L[i - 1] + 4 * L[i] - 2 * L[i + 1] +
        L[i + w - 1] - 2 * L[i + w] + L[i + w + 1];
      sum += Math.abs(v);
      count++;
    }
  }
  return count ? (Math.sqrt(Math.PI / 2) * sum) / (6 * count) : 0;
}

// Ratio of luma steps across the 8x8 JPEG grid vs. inside blocks. ~1 = clean.
export function blockiness(L, w, h) {
  let edge = 0, edgeN = 0, inner = 0, innerN = 0;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w - 1; x++) {
      const dd = Math.abs(L[row + x + 1] - L[row + x]);
      if ((x + 1) % 8 === 0) { edge += dd; edgeN++; } else { inner += dd; innerN++; }
    }
  }
  for (let y = 0; y < h - 1; y++) {
    const isEdge = (y + 1) % 8 === 0;
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const dd = Math.abs(L[row + w + x] - L[row + x]);
      if (isEdge) { edge += dd; edgeN++; } else { inner += dd; innerN++; }
    }
  }
  if (!edgeN || !innerN) return 1;
  return (edge / edgeN + 0.5) / (inner / innerN + 0.5);
}

// Centre crop (aligned to the 8px JPEG grid) of at most size x size pixels.
// Share of 4x4 blocks holding exactly one value (clipped black and white are
// ignored). Drawn and digital art is full of perfectly flat fills; camera
// photos almost never are, because of sensor noise and dithering.
export function flatShare(L, w, h) {
  let flat = 0, total = 0;
  for (let y = 0; y + 4 <= h; y += 4) {
    for (let x = 0; x + 4 <= w; x += 4) {
      const v = L[y * w + x];
      if (v < 8 || v > 247) continue;
      total++;
      let same = true;
      for (let j = 0; j < 4 && same; j++) {
        for (let i = 0; i < 4; i++) {
          if (L[(y + j) * w + x + i] !== v) {
            same = false;
            break;
          }
        }
      }
      if (same) flat++;
    }
  }
  return total ? flat / total : 0;
}

function centerCrop(L, w, h, size) {
  if (w <= size && h <= size) return { L, w, h };
  const cw = Math.min(w, size), ch = Math.min(h, size);
  const x0 = Math.floor((w - cw) / 16) * 8, y0 = Math.floor((h - ch) / 16) * 8;
  const out = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) out.set(L.subarray((y0 + y) * w + x0, (y0 + y) * w + x0 + cw), y * cw);
  return { L: out, w: cw, h: ch };
}

// Distinct opaque colours, counting stops once `limit` is exceeded.
export function countColors(img, limit = 4096) {
  const d = img.data;
  const seen = new Set();
  for (let p = 0; p < d.length; p += 4) {
    if (d[p + 3] < 8) continue;
    seen.add((d[p] << 16) | (d[p + 1] << 8) | d[p + 2]);
    if (seen.size > limit) return limit + 1;
  }
  return seen.size;
}

const RATIOS = [[1, 1], [4, 3], [3, 2], [16, 9], [16, 10], [5, 4], [21, 9], [3, 4], [2, 3], [9, 16], [4, 5]];
export function aspectLabel(w, h) {
  const r = w / h;
  for (const [a, b] of RATIOS) if (Math.abs(r / (a / b) - 1) < 0.01) return `${a}:${b}`;
  return r >= 1 ? `${r.toFixed(2)}:1` : `1:${(1 / r).toFixed(2)}`;
}

function resolutionTier(w, h) {
  const long = Math.max(w, h);
  if (long >= 7680) return '8K+';
  if (long >= 3840) return '4K';
  if (long >= 2560) return 'QHD';
  if (long >= 1920) return 'Full HD';
  if (long >= 1280) return 'HD';
  if (long >= 640) return 'Low';
  return 'Very low';
}

function colorStats(img) {
  const d = img.data;
  const hist = { r: new Uint32Array(256), g: new Uint32Array(256), b: new Uint32Array(256), l: new Uint32Array(256) };
  let n = 0, satSum = 0, rgS = 0, rgSq = 0, ybS = 0, ybSq = 0;
  let transparent = 0;
  for (let p = 0; p < d.length; p += 4) {
    if (d[p + 3] < 8) { transparent++; continue; }
    const r = d[p], g = d[p + 1], b = d[p + 2];
    hist.r[r]++; hist.g[g]++; hist.b[b]++;
    const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    hist.l[Math.round(y)]++;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    satSum += max > 0 ? (max - min) / max : 0;
    const rg = r - g, yb = 0.5 * (r + g) - b;
    rgS += rg; rgSq += rg * rg; ybS += yb; ybSq += yb * yb;
    n++;
  }
  const total = d.length / 4;
  if (!n) return { hist, n, transparentPct: 1, saturation: 0, colorfulness: 0, cast: { label: 'Neutral', strength: 0 } };
  const muRg = rgS / n, muYb = ybS / n;
  const sdRg = Math.sqrt(Math.max(0, rgSq / n - muRg * muRg));
  const sdYb = Math.sqrt(Math.max(0, ybSq / n - muYb * muYb));
  const colorfulness = Math.hypot(sdRg, sdYb) + 0.3 * Math.hypot(muRg, muYb);
  let cast = { label: 'Neutral', strength: 0 };
  const neutral = neutralMean(img);
  if (neutral) {
    const [ar, ag, ab] = neutral;
    const m = (ar + ag + ab) / 3;
    const rb = (ar - ab) / m, gm = (ag - (ar + ab) / 2) / m;
    const strength = Math.max(Math.abs(rb), Math.abs(gm));
    if (strength > 0.08) {
      const label = Math.abs(rb) >= Math.abs(gm) ? (rb > 0 ? 'Warm' : 'Cool') : gm > 0 ? 'Green' : 'Magenta';
      cast = { label, strength: Math.round(strength * 100) / 100 };
    }
  }
  return { hist, n, transparentPct: transparent / total, saturation: satSum / n, colorfulness, cast };
}

// detailImage (optional) is the same picture at a reference size. Noise,
// sharpness and blockiness are scale-dependent, so an enhanced result is
// measured at the original's size for a fair before/after comparison.
export function analyzeImage(img, { detailImage = img } = {}) {
  const { width: w, height: h } = img;
  const alpha = hasTransparency(img);
  const color = colorStats(img);
  const { hist, n } = color;

  let lumSum = 0, lumSq = 0;
  for (let v = 0; v < 256; v++) { lumSum += v * hist.l[v]; lumSq += v * v * hist.l[v]; }
  const brightness = n ? lumSum / n : 0;
  const contrast = n ? Math.sqrt(Math.max(0, lumSq / n - brightness * brightness)) : 0;
  let dark = 0, bright = 0, paper = 0, ink = 0;
  for (let v = 0; v < 256; v++) {
    if (v <= 6) dark += hist.l[v];
    if (v >= 249) bright += hist.l[v];
    if (v >= 220) paper += hist.l[v];
    if (v <= 140) ink += hist.l[v];
  }
  const shadowClip = n ? dark / n : 0;
  const highlightClip = n ? bright / n : 0;
  const p1 = n ? percentile(hist.l, n, 0.01) : 0;
  const p99 = n ? percentile(hist.l, n, 0.99) : 0;

  // A handful of flat colours cannot contain sensor noise or JPEG artifacts,
  // and the noise estimators misread hard pixel edges as grain.
  const det = detailImage;
  const dw = det.width, dh = det.height, dLong = Math.max(dw, dh);
  const uniqueColors = countColors(img);
  const flatColors = uniqueColors <= 64;
  const isPixelArt = dLong <= 512 && flatColors;

  const fullL = opaqueLuma(det);
  const crop = centerCrop(fullL, dw, dh, 1536);
  const noiseSigma = flatColors ? 0 : estimateGrain(crop.L, crop.w, crop.h);
  const block = flatColors ? 1 : blockiness(crop.L, crop.w, crop.h);

  const small = dLong > 1024 ? resize(det, (dw * 1024) / dLong, (dh * 1024) / dLong) : det;
  const smallL = small === det ? fullL : opaqueLuma(small);
  const lapVar = laplacianVariance(smallL, small.width, small.height);
  const edge = edgeWidth(smallL, small.width, small.height);
  // The same measure at full resolution, which tells a blown-up picture
  // (edges blurred over many pixels) from a large photo that is merely soft.
  const nativeEdge = small === det ? edge : edgeWidth(crop.L, crop.w, crop.h);

  const isGrayscale = color.colorfulness < 3 && color.saturation < 0.04;
  const isDocument = n > 0 && paper / n > 0.45 && ink / n > 0.005 && ink / n < 0.4 && color.colorfulness < 20;
  // Clean, uncompressed and mostly flat-filled: an illustration, anime frame
  // or graphic rather than a photo.
  const isIllustration = !isPixelArt && !isDocument && noiseSigma < 1 && block < 1.15 && flatShare(crop.L, crop.w, crop.h) > 0.35;
  const megapixels = (w * h) / 1e6;

  const sharpScore = isPixelArt ? 100 : edge == null ? 70 : score((9 - edge) / 5.8);
  const noiseScore = score(1 - (noiseSigma - 1.5) / 13.5);
  const blockScore = score(1 - (block - 1.1) / 0.7);
  const resScore = score(Math.log2(megapixels / 0.08) / Math.log2(100));
  const clipPenalty = shadowClip * 1.5 + highlightClip * 2;
  const expScore = isDocument
    ? score(1 - shadowClip * 1.5)
    : score(1 - Math.max(0, Math.abs(brightness - 118) - 35) / 80 - clipPenalty);
  const conScore = isDocument
    ? 100
    : score(contrast < 45 ? 1 - (45 - contrast) / 40 : 1 - Math.max(0, contrast - 80) / 60);

  let exposure = 'Balanced';
  if (!isDocument) {
    if (brightness < 70 || (shadowClip > 0.15 && brightness < 100)) exposure = 'Underexposed';
    else if (brightness > 190 || highlightClip > 0.1) exposure = 'Overexposed';
  }

  // Quality is limited by the weakest link, so the worst key factor pulls the
  // weighted average down.
  const base = 0.3 * sharpScore + 0.2 * noiseScore + 0.15 * expScore + 0.1 * conScore + 0.15 * resScore + 0.1 * blockScore;
  const worst = Math.min(sharpScore, noiseScore, expScore, resScore, blockScore);
  const overall = Math.round(0.65 * base + 0.35 * worst);
  const grade = overall >= 85 ? 'Excellent' : overall >= 70 ? 'Good' : overall >= 50 ? 'Fair' : 'Poor';

  return {
    width: w,
    height: h,
    megapixels,
    aspect: aspectLabel(w, h),
    hasAlpha: alpha,
    transparentPct: color.transparentPct,
    histogram: hist,
    brightness,
    contrast,
    shadowClip,
    highlightClip,
    dynamicRange: p99 - p1,
    saturation: color.saturation,
    colorfulness: color.colorfulness,
    isGrayscale,
    isDocument,
    isPixelArt,
    isIllustration,
    uniqueColors,
    cast: color.cast,
    exposure: { label: exposure, score: expScore },
    contrastScore: conScore,
    sharpness: { variance: lapVar, edgeWidth: edge, nativeEdgeWidth: nativeEdge, score: sharpScore },
    noise: { sigma: noiseSigma, score: noiseScore },
    blockiness: { ratio: block, score: blockScore },
    resolution: { score: resScore, tier: resolutionTier(w, h), printInches: [w / 300, h / 300] },
    palette: dominantColors(img),
    score: overall,
    grade,
  };
}

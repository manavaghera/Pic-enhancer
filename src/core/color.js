// Global colour and tone corrections: automatic levels + white balance, and
// manual brightness / contrast / warmth / saturation / vibrance.

import { cloneImage } from './image.js';

const MAX_STRETCH = 2;

export function percentile(hist, count, q) {
  const target = q * count;
  let acc = 0;
  for (let i = 0; i < hist.length; i++) {
    acc += hist[i];
    if (acc > target) return i;
  }
  return hist.length - 1;
}

const tintOf = ([r, g, b]) => {
  const m = (r + g + b) / 3;
  return [(r - b) / m, (g - (r + b) / 2) / m];
};

// Mean colour of near-neutral, well-exposed pixels. These should be grey, so a
// tint they share is a colour cast. Strongly coloured pixels (blue sky, green
// grass) are ignored. A real cast tints shadows, midtones and highlights alike,
// so the tint must agree across tonal bands and include the highlights; a tint
// confined to one band (a purple mountain range, say) only counts when it
// covers a large part of the image. Returns null when there is not enough
// consistent evidence to judge.
export function neutralMean(img) {
  const d = img.data;
  const bands = [0, 1, 2].map(() => ({ r: 0, g: 0, b: 0, n: 0 }));
  let total = 0;
  for (let p = 0; p < d.length; p += 4) {
    if (d[p + 3] < 8) continue;
    total++;
    const R = d[p], G = d[p + 1], B = d[p + 2];
    const max = Math.max(R, G, B);
    const min = Math.min(R, G, B);
    if (max < 40 || max > 245 || (max - min) / max > 0.2) continue;
    const band = bands[max < 100 ? 0 : max <= 180 ? 1 : 2];
    band.r += R;
    band.g += G;
    band.b += B;
    band.n++;
  }
  const all = bands.reduce((acc, bd) => ({ r: acc.r + bd.r, g: acc.g + bd.g, b: acc.b + bd.b, n: acc.n + bd.n }));
  if (all.n < 64 || all.n < total * 0.02 || all.r === 0 || all.g === 0 || all.b === 0) return null;
  const mean = [all.r / all.n, all.g / all.n, all.b / all.n];
  const valid = bands.filter((bd) => bd.n >= Math.max(32, total * 0.005));
  if (valid.length === 0) return null;
  if (valid.length === 1) return valid[0].n >= total * 0.25 ? mean : null;
  // Highlights are the most reliable reference ("white patch"); without
  // near-neutral highlights, darker greys are too often just muted colours.
  if (!valid.includes(bands[2])) return null;
  const [orb, ogm] = tintOf(mean);
  const consistent = valid.every((bd) => {
    const [rb, gm] = tintOf([bd.r / bd.n, bd.g / bd.n, bd.b / bd.n]);
    return rb * orb + gm * ogm > 0;
  });
  return consistent ? mean : null;
}

// Per-channel gains that turn the neutral mean grey (clamped to stay subtle).
export function whiteBalanceGains(img) {
  const m = neutralMean(img);
  if (!m) return [1, 1, 1];
  const mean = (m[0] + m[1] + m[2]) / 3;
  return m.map((c) => Math.min(1.2, Math.max(0.84, mean / c)));
}

// Neutralises colour casts, then stretches all channels together (ignoring
// 0.4% outliers each side) so contrast improves without shifting hues.
// strength 0..1 blends from no change to the full fix.
export function autoTone(img, strength = 1) {
  if (!(strength > 0)) return cloneImage(img);
  const s = Math.min(1, strength);
  const gains = whiteBalanceGains(img).map((gn) => 1 + (gn - 1) * s);
  const hist = new Uint32Array(256);
  const src = img.data;
  let count = 0;
  for (let p = 0; p < src.length; p += 4) {
    if (src[p + 3] < 8) continue;
    for (let c = 0; c < 3; c++) hist[Math.min(255, Math.round(src[p + c] * gains[c]))]++;
    count += 3;
  }
  let lo = 0;
  let hi = 255;
  if (count > 0) {
    lo = percentile(hist, count, 0.004);
    hi = percentile(hist, count, 0.996);
  }
  if (hi - lo < 24) {
    lo = 0;
    hi = 255;
  }
  // Limit the stretch so hazy or high-key images are improved, not blown out.
  const minRange = 255 / MAX_STRETCH;
  if (hi - lo < minRange) {
    lo = Math.min(255 - minRange, Math.max(0, (lo + hi) / 2 - minRange / 2));
    hi = lo + minRange;
  }
  const luts = gains.map((gain) => {
    const lut = new Uint8ClampedArray(256);
    for (let v = 0; v < 256; v++) {
      const stretched = ((v * gain - lo) * 255) / (hi - lo);
      lut[v] = v + (stretched - v) * s;
    }
    return lut;
  });
  const out = cloneImage(img);
  const d = out.data;
  const [lr, lg, lb] = luts;
  for (let p = 0; p < d.length; p += 4) {
    d[p] = lr[d[p]];
    d[p + 1] = lg[d[p + 1]];
    d[p + 2] = lb[d[p + 2]];
  }
  return out;
}

// All parameters are -1..1; zero means "no change".
export function adjustTone(img, { brightness = 0, contrast = 0, warmth = 0, saturation = 0, vibrance = 0 } = {}) {
  const identityTone = brightness === 0 && contrast === 0 && warmth === 0;
  const identityColor = saturation === 0 && vibrance === 0;
  if (identityTone && identityColor) return cloneImage(img);

  const gamma = Math.pow(2, -brightness);
  const gains = [1 + 0.12 * warmth, 1 + 0.02 * warmth, 1 - 0.12 * warmth];
  const luts = gains.map((gain) => {
    const lut = new Float32Array(256);
    for (let v = 0; v < 256; v++) {
      let x = Math.min(1, (v * gain) / 255);
      x = Math.pow(x, gamma);
      if (contrast > 0) {
        const smooth = x * x * (3 - 2 * x);
        x += (smooth - x) * contrast;
      } else if (contrast < 0) {
        x += (0.5 - x) * -contrast * 0.5;
      }
      lut[v] = x * 255;
    }
    return lut;
  });

  const out = cloneImage(img);
  const d = out.data;
  const [lr, lg, lb] = luts;
  const satGain = 1 + saturation;
  for (let p = 0; p < d.length; p += 4) {
    let r = lr[d[p]];
    let g = lg[d[p + 1]];
    let b = lb[d[p + 2]];
    if (!identityColor) {
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const sat = max > 0 ? (max - min) / max : 0;
      const f = satGain * (1 + vibrance * (1 - sat));
      const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      r = y + (r - y) * f;
      g = y + (g - y) * f;
      b = y + (b - y) * f;
    }
    d[p] = r;
    d[p + 1] = g;
    d[p + 2] = b;
  }
  return out;
}

// Plain RGBA image helpers. An "image" is { width, height, data: Uint8ClampedArray }
// so the same code runs in the worker, the main thread and in Node tests.

export function createImage(width, height) {
  return { width, height, data: new Uint8ClampedArray(width * height * 4) };
}

export function cloneImage(img) {
  return { width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) };
}

// Rec. 709 luma on gamma-encoded values; good enough for metrics and sharpening.
export function luma(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function toLuma(img) {
  const { data } = img;
  const n = img.width * img.height;
  const out = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    out[i] = 0.2126 * data[p] + 0.7152 * data[p + 1] + 0.0722 * data[p + 2];
  }
  return out;
}

export function hasTransparency(img) {
  const { data } = img;
  for (let p = 3; p < data.length; p += 4) {
    if (data[p] < 255) return true;
  }
  return false;
}

export function premultiply(img) {
  const d = img.data;
  for (let p = 0; p < d.length; p += 4) {
    const a = d[p + 3];
    if (a === 255) continue;
    const k = a / 255;
    d[p] = d[p] * k;
    d[p + 1] = d[p + 1] * k;
    d[p + 2] = d[p + 2] * k;
  }
}

export function unpremultiply(img) {
  const d = img.data;
  for (let p = 0; p < d.length; p += 4) {
    const a = d[p + 3];
    if (a === 255) continue;
    if (a === 0) {
      d[p] = d[p + 1] = d[p + 2] = 0;
      continue;
    }
    const k = 255 / a;
    d[p] = d[p] * k;
    d[p + 1] = d[p + 1] * k;
    d[p + 2] = d[p + 2] * k;
  }
}

// Gives fully transparent pixels the colour of nearby opaque ones, so filters
// that ignore alpha (the AI model) do not drag hidden black into soft edges.
export function bleedTransparent(img, passes = 6) {
  const { width: w, height: h } = img;
  const out = cloneImage(img);
  const d = out.data;
  let known = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) known[i] = d[i * 4 + 3] >= 8 ? 1 : 0;
  for (let pass = 0; pass < passes; pass++) {
    const next = known.slice();
    let changed = false;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (known[i]) continue;
        let r = 0, g = 0, b = 0, c = 0;
        for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
          if (j < 0 || !known[j]) continue;
          r += d[j * 4];
          g += d[j * 4 + 1];
          b += d[j * 4 + 2];
          c++;
        }
        if (!c) continue;
        d[i * 4] = r / c;
        d[i * 4 + 1] = g / c;
        d[i * 4 + 2] = b / c;
        next[i] = 1;
        changed = true;
      }
    }
    known = next;
    if (!changed) break;
  }
  return out;
}

// Runs fn over overlapping horizontal bands so large images never need
// full-size float buffers. fn receives a read-only band view and must return
// a new Uint8ClampedArray of the same size. Rows inside `overlap` of an inner
// band edge are discarded, so any filter with support <= overlap is exact.
export function mapBands(img, overlap, fn, onProgress, targetPixels = 1 << 21) {
  const { width: w, height: h } = img;
  const bandRows = Math.max(32, Math.floor(targetPixels / Math.max(1, w)));
  if (bandRows >= h) {
    return { width: w, height: h, data: fn(img) };
  }
  const out = new Uint8ClampedArray(img.data.length);
  for (let y0 = 0; y0 < h; y0 += bandRows) {
    const y1 = Math.min(h, y0 + bandRows);
    const top = Math.max(0, y0 - overlap);
    const bottom = Math.min(h, y1 + overlap);
    const band = {
      width: w,
      height: bottom - top,
      data: img.data.subarray(top * w * 4, bottom * w * 4),
    };
    const res = fn(band);
    out.set(res.subarray((y0 - top) * w * 4, (y1 - top) * w * 4), y0 * w * 4);
    if (onProgress) onProgress(y1 / h);
  }
  return { width: w, height: h, data: out };
}

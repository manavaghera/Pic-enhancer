// Deterministic synthetic images for tests.

export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand) {
  const u = Math.max(rand(), 1e-12);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function fromFn(w, h, fn) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b, a = 255] = fn(x, y);
      const p = (y * w + x) * 4;
      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;
      data[p + 3] = a;
    }
  }
  return { width: w, height: h, data };
}

export const solid = (w, h, rgba) => fromFn(w, h, () => rgba);

export function addNoise(img, sigma, seed = 7, mono = true) {
  const rand = rng(seed);
  const d = new Uint8ClampedArray(img.data);
  for (let p = 0; p < d.length; p += 4) {
    const n = gaussian(rand) * sigma;
    d[p] += n;
    d[p + 1] += mono ? n : gaussian(rand) * sigma;
    d[p + 2] += mono ? n : gaussian(rand) * sigma;
  }
  return { width: img.width, height: img.height, data: d };
}

export function checker(w, h, size, lo = 40, hi = 210) {
  return fromFn(w, h, (x, y) => {
    const v = (Math.floor(x / size) + Math.floor(y / size)) % 2 ? hi : lo;
    return [v, v, v];
  });
}

export function channelStats(img, ch = 0, x0 = 0, y0 = 0, x1 = img.width, y1 = img.height) {
  let sum = 0, sq = 0, n = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const v = img.data[(y * img.width + x) * 4 + ch];
      sum += v;
      sq += v * v;
      n++;
    }
  }
  const mean = sum / n;
  return { mean, std: Math.sqrt(Math.max(0, sq / n - mean * mean)) };
}

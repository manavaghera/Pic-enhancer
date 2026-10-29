// Dominant colours via k-means++ on a deterministic pixel sample.

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const toHex = (v) => Math.round(v).toString(16).padStart(2, '0');

export function dominantColors(img, k = 6, maxSamples = 12000) {
  const d = img.data;
  const total = img.width * img.height;
  const step = Math.max(1, Math.floor(total / maxSamples));
  const samples = [];
  for (let i = 0; i < total; i += step) {
    const p = i * 4;
    if (d[p + 3] < 128) continue;
    samples.push(d[p], d[p + 1], d[p + 2]);
  }
  const n = samples.length / 3;
  if (n === 0) return [];
  const kk = Math.min(k, n);
  const rand = mulberry32(0x9e3779b9);
  const centers = new Float64Array(kk * 3);
  const dist = new Float64Array(n).fill(Infinity);

  const first = Math.floor(rand() * n);
  centers.set(samples.slice(first * 3, first * 3 + 3), 0);
  for (let c = 1; c < kk; c++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const dr = samples[i * 3] - centers[(c - 1) * 3];
      const dg = samples[i * 3 + 1] - centers[(c - 1) * 3 + 1];
      const db = samples[i * 3 + 2] - centers[(c - 1) * 3 + 2];
      const dd = dr * dr + dg * dg + db * db;
      if (dd < dist[i]) dist[i] = dd;
      sum += dist[i];
    }
    let pick = 0;
    if (sum > 0) {
      let target = rand() * sum;
      for (let i = 0; i < n; i++) {
        target -= dist[i];
        if (target <= 0) {
          pick = i;
          break;
        }
      }
    }
    centers.set(samples.slice(pick * 3, pick * 3 + 3), c * 3);
  }

  const assign = new Int32Array(n);
  const counts = new Float64Array(kk);
  for (let iter = 0; iter < 12; iter++) {
    const sums = new Float64Array(kk * 3);
    counts.fill(0);
    let moved = 0;
    for (let i = 0; i < n; i++) {
      const r = samples[i * 3], g = samples[i * 3 + 1], b = samples[i * 3 + 2];
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < kk; c++) {
        const dr = r - centers[c * 3], dg = g - centers[c * 3 + 1], db = b - centers[c * 3 + 2];
        const dd = dr * dr + dg * dg + db * db;
        if (dd < bestD) {
          bestD = dd;
          best = c;
        }
      }
      if (assign[i] !== best) moved++;
      assign[i] = best;
      counts[best]++;
      sums[best * 3] += r;
      sums[best * 3 + 1] += g;
      sums[best * 3 + 2] += b;
    }
    for (let c = 0; c < kk; c++) {
      if (counts[c] === 0) continue;
      centers[c * 3] = sums[c * 3] / counts[c];
      centers[c * 3 + 1] = sums[c * 3 + 1] / counts[c];
      centers[c * 3 + 2] = sums[c * 3 + 2] / counts[c];
    }
    if (iter > 0 && moved === 0) break;
  }

  const result = [];
  for (let c = 0; c < kk; c++) {
    if (counts[c] === 0) continue;
    const rgb = [centers[c * 3], centers[c * 3 + 1], centers[c * 3 + 2]].map((v) => Math.round(v));
    result.push({ rgb, hex: `#${rgb.map(toHex).join('')}`, pct: counts[c] / n });
  }
  return result.sort((a, b) => b.pct - a.pct);
}

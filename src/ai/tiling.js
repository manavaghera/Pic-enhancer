// Tiled super-resolution for models with a fixed input size. Tiles overlap so
// every output pixel is produced with at least `pad` pixels of context; each
// tile owns the part of the overlap up to its midpoint (hard cut, as in the
// reference Real-ESRGAN tile mode), so no full-size float buffers are needed.

// Tile origins along one axis; the last tile sits flush with the far edge.
export function tileStarts(len, tile, pad) {
  if (len <= tile) return [0];
  const step = Math.max(1, tile - 2 * pad);
  const starts = [];
  for (let p = 0; ; p += step) {
    if (p + tile >= len) {
      starts.push(len - tile);
      return starts;
    }
    starts.push(p);
  }
}

// [from, to) range of the axis each tile writes, split at overlap midpoints.
export function tileSpans(starts, tile, len) {
  return starts.map((start, i) => ({
    start,
    from: i === 0 ? 0 : Math.floor((starts[i - 1] + tile + start) / 2),
    to: i === starts.length - 1 ? len : Math.floor((start + tile + starts[i + 1]) / 2),
  }));
}

// runTile(Float32Array tile*tile*3 RGB in 0..1) -> Promise<Float32Array of
// (tile*scale)^2*3 RGB in 0..1>. Returns an opaque RGBA image scale times larger.
export async function upscaleTiled(img, { tile, scale, pad }, runTile, onProgress) {
  const { width: w, height: h, data } = img;
  const xs = tileSpans(tileStarts(Math.max(w, tile), tile, pad), tile, Math.max(w, tile));
  const ys = tileSpans(tileStarts(Math.max(h, tile), tile, pad), tile, Math.max(h, tile));
  const ow = w * scale;
  const out = new Uint8ClampedArray(ow * h * scale * 4);
  const input = new Float32Array(tile * tile * 3);
  const span = tile * scale;
  const total = xs.length * ys.length;
  let done = 0;

  for (const ty of ys) {
    for (const tx of xs) {
      // Positions past a small image's edge repeat the last row/column.
      for (let y = 0; y < tile; y++) {
        const row = Math.min(h - 1, ty.start + y) * w;
        for (let x = 0; x < tile; x++) {
          const p = (row + Math.min(w - 1, tx.start + x)) * 4;
          const q = (y * tile + x) * 3;
          input[q] = data[p] / 255;
          input[q + 1] = data[p + 1] / 255;
          input[q + 2] = data[p + 2] / 255;
        }
      }
      const result = await runTile(input);
      const x0 = tx.from * scale, x1 = Math.min(tx.to, w) * scale;
      const y0 = ty.from * scale, y1 = Math.min(ty.to, h) * scale;
      for (let oy = y0; oy < y1; oy++) {
        const src = (oy - ty.start * scale) * span - tx.start * scale;
        for (let ox = x0; ox < x1; ox++) {
          const r = (src + ox) * 3;
          const o = (oy * ow + ox) * 4;
          out[o] = result[r] * 255;
          out[o + 1] = result[r + 1] * 255;
          out[o + 2] = result[r + 2] * 255;
          out[o + 3] = 255;
        }
      }
      onProgress?.(++done / total);
    }
  }
  return { width: ow, height: h * scale, data: out };
}

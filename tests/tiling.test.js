import { describe, expect, it } from 'vitest';
import { tileSpans, tileStarts, upscaleTiled } from '../src/ai/tiling.js';
import { bleedTransparent } from '../src/core/image.js';
import { resize } from '../src/core/resize.js';
import { fromFn, rng } from './helpers.js';

// Fake 4x model: nearest-neighbour on the RGB tile, so a correct tiler must
// reproduce a whole-image nearest-neighbour upscale exactly.
const nearestTile = (tile, scale) => async (input) => {
  const span = tile * scale;
  const out = new Float32Array(span * span * 3);
  for (let y = 0; y < span; y++) {
    for (let x = 0; x < span; x++) {
      const q = (Math.floor(y / scale) * tile + Math.floor(x / scale)) * 3;
      out.set(input.subarray(q, q + 3), (y * span + x) * 3);
    }
  }
  return out;
};

describe('tile layout', () => {
  it('covers each axis with overlapping tiles, the last flush with the edge', () => {
    expect(tileStarts(64, 64, 10)).toEqual([0]);
    expect(tileStarts(30, 64, 10)).toEqual([0]);
    expect(tileStarts(100, 64, 10)).toEqual([0, 36]);
    expect(tileStarts(200, 64, 10)).toEqual([0, 44, 88, 132, 136]);
  });

  it('splits ownership at overlap midpoints with at least pad pixels of context', () => {
    for (const len of [65, 100, 137, 200, 513]) {
      const starts = tileStarts(len, 64, 10);
      const spans = tileSpans(starts, 64, len);
      expect(spans[0].from).toBe(0);
      expect(spans.at(-1).to).toBe(len);
      spans.forEach((s, i) => {
        if (i > 0) expect(s.from).toBe(spans[i - 1].to);
        expect(s.from - s.start === 0 || s.from - s.start >= 10).toBe(true);
        expect(s.start + 64 - s.to === 0 || s.start + 64 - s.to >= 10).toBe(true);
      });
    }
  });
});

describe('upscaleTiled', () => {
  it('reproduces a whole-image upscale exactly for a local model', async () => {
    const rand = rng(2);
    for (const [w, h] of [[150, 97], [64, 64], [23, 70], [5, 3]]) {
      const img = fromFn(w, h, () => [rand() * 255, rand() * 255, rand() * 255]);
      const out = await upscaleTiled(img, { tile: 64, scale: 4, pad: 10 }, nearestTile(64, 4));
      const ref = resize(img, w * 4, h * 4, { method: 'nearest' });
      expect(out.width).toBe(w * 4);
      expect(out.height).toBe(h * 4);
      expect(Buffer.from(out.data).equals(Buffer.from(ref.data))).toBe(true);
    }
  });

  it('writes every output pixel and reports progress to 1', async () => {
    const img = fromFn(130, 90, (x, y) => [x, y, 0]);
    const seen = [];
    const out = await upscaleTiled(img, { tile: 64, scale: 2, pad: 8 }, nearestTile(64, 2), (f) => seen.push(f));
    for (let p = 3; p < out.data.length; p += 4) expect(out.data[p]).toBe(255);
    expect(seen.at(-1)).toBe(1);
  });
});

describe('bleedTransparent', () => {
  it('fills hidden pixels near opaque ones and leaves alpha alone', () => {
    const img = fromFn(10, 1, (x) => (x < 3 ? [0, 0, 200, 255] : [255, 0, 0, 0]));
    const out = bleedTransparent(img, 4);
    expect(Array.from(out.data.slice(3 * 4, 3 * 4 + 4))).toEqual([0, 0, 200, 0]);
    expect(Array.from(out.data.slice(6 * 4, 6 * 4 + 3))).toEqual([0, 0, 200]);
    expect(Array.from(out.data.slice(9 * 4, 9 * 4 + 3))).toEqual([255, 0, 0]);
  });
});

import { describe, expect, it } from 'vitest';
import { buildWeights, resize } from '../src/core/resize.js';
import { channelStats, fromFn, solid } from './helpers.js';

describe('buildWeights', () => {
  it('normalises every output tap set to 1', () => {
    for (const [src, dst] of [[10, 37], [100, 13], [1, 4], [7, 7]]) {
      const { counts, weights, maxTaps } = buildWeights(src, dst);
      for (let i = 0; i < dst; i++) {
        let sum = 0;
        for (let k = 0; k < counts[i]; k++) sum += weights[i * maxTaps + k];
        expect(sum).toBeCloseTo(1, 5);
      }
    }
  });
});

describe('resize', () => {
  it('produces the requested size', () => {
    const out = resize(solid(13, 7, [10, 20, 30, 255]), 52, 28);
    expect(out.width).toBe(52);
    expect(out.height).toBe(28);
    expect(out.data.length).toBe(52 * 28 * 4);
  });

  it('keeps a flat colour flat when upscaling and downscaling', () => {
    const img = solid(20, 15, [90, 140, 200, 255]);
    for (const [w, h] of [[80, 60], [7, 5], [33, 41]]) {
      const out = resize(img, w, h);
      for (let p = 0; p < out.data.length; p += 4) {
        expect(Math.abs(out.data[p] - 90)).toBeLessThanOrEqual(1);
        expect(Math.abs(out.data[p + 2] - 200)).toBeLessThanOrEqual(1);
        expect(out.data[p + 3]).toBe(255);
      }
    }
  });

  it('upscales a single pixel', () => {
    const out = resize(solid(1, 1, [5, 6, 7, 255]), 5, 3);
    expect(Array.from(out.data.slice(0, 4))).toEqual([5, 6, 7, 255]);
  });

  it('nearest-neighbour replicates pixels exactly', () => {
    const img = fromFn(2, 2, (x, y) => [x * 100, y * 100, 50]);
    const out = resize(img, 4, 4, { method: 'nearest' });
    const px = (x, y) => Array.from(out.data.slice((y * 4 + x) * 4, (y * 4 + x) * 4 + 3));
    expect(px(0, 0)).toEqual([0, 0, 50]);
    expect(px(1, 1)).toEqual([0, 0, 50]);
    expect(px(2, 0)).toEqual([100, 0, 50]);
    expect(px(3, 3)).toEqual([100, 100, 50]);
  });

  it('does not bleed the colour of fully transparent pixels', () => {
    const img = fromFn(8, 8, (x) => (x < 4 ? [0, 0, 255, 255] : [255, 0, 0, 0]));
    const out = resize(img, 32, 32);
    for (let p = 0; p < out.data.length; p += 4) {
      if (out.data[p + 3] > 0) expect(out.data[p]).toBeLessThan(20);
    }
  });

  it('preserves average brightness of a gradient', () => {
    const img = fromFn(40, 30, (x) => [x * 6, x * 6, x * 6]);
    const out = resize(img, 160, 120);
    expect(Math.abs(channelStats(out).mean - channelStats(img).mean)).toBeLessThan(1.5);
  });

  it('reports progress up to 1', () => {
    const seen = [];
    resize(solid(64, 64, [1, 2, 3, 255]), 256, 256, { onProgress: (f) => seen.push(f) });
    expect(seen.at(-1)).toBe(1);
    for (let i = 1; i < seen.length; i++) expect(seen[i]).toBeGreaterThanOrEqual(seen[i - 1]);
  });
});

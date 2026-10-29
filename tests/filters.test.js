import { describe, expect, it } from 'vitest';
import { boxBlur, clarity, denoise, gaussianBlur, gaussianSupport, sharpen } from '../src/core/filters.js';
import { mapBands, toLuma } from '../src/core/image.js';
import { laplacianVariance } from '../src/core/analyze.js';
import { addNoise, channelStats, checker, fromFn, solid } from './helpers.js';

describe('blurs', () => {
  it('box blur keeps constants and mass', () => {
    const w = 21, h = 21;
    const flat = new Float32Array(w * h).fill(7);
    for (const v of boxBlur(flat, w, h, 3)) expect(v).toBeCloseTo(7, 4);
    const impulse = new Float32Array(w * h);
    impulse[10 * w + 10] = 100;
    const out = boxBlur(impulse, w, h, 2);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(100, 3);
    expect(out[10 * w + 10]).toBeCloseTo(4, 4);
  });

  it('gaussian blur keeps constants and mass for small and large sigma', () => {
    const w = 61, h = 61;
    for (const sigma of [0.8, 1.7, 5, 9]) {
      const flat = new Float32Array(w * h).fill(50);
      for (const v of gaussianBlur(flat, w, h, sigma)) expect(v).toBeCloseTo(50, 3);
      const impulse = new Float32Array(w * h);
      impulse[30 * w + 30] = 1000;
      const sum = gaussianBlur(impulse, w, h, sigma).reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1000, 1);
      expect(gaussianSupport(sigma)).toBeGreaterThan(0);
    }
  });
});

describe('mapBands', () => {
  it('matches whole-image processing when bands overlap enough', () => {
    const img = addNoise(checker(37, 300, 9), 20);
    const sigma = 2;
    const fn = (band) => {
      const blurred = gaussianBlur(toLuma(band), band.width, band.height, sigma);
      const out = new Uint8ClampedArray(band.data.length);
      for (let i = 0; i < blurred.length; i++) out.fill(blurred[i], i * 4, i * 4 + 4);
      return out;
    };
    const whole = fn(img);
    const banded = mapBands(img, gaussianSupport(sigma) + 2, fn, null, 37 * 40).data;
    for (let i = 0; i < whole.length; i++) expect(Math.abs(whole[i] - banded[i])).toBeLessThanOrEqual(1);
  });
});

describe('denoise', () => {
  it('removes most noise from a flat area', () => {
    const noisy = addNoise(solid(96, 96, [128, 128, 128, 255]), 12, 3, false);
    const before = channelStats(noisy, 1).std;
    const after = channelStats(denoise(noisy, 0.7), 1).std;
    expect(after).toBeLessThan(before * 0.5);
  });

  it('keeps a strong edge', () => {
    const step = fromFn(96, 64, (x) => (x < 48 ? [50, 50, 50] : [200, 200, 200]));
    const out = denoise(addNoise(step, 10), 0.8);
    const left = channelStats(out, 0, 30, 0, 44, 64).mean;
    const right = channelStats(out, 0, 52, 0, 66, 64).mean;
    expect(right - left).toBeGreaterThan(140);
  });

  it('is a no-op at strength 0 and keeps alpha', () => {
    const img = fromFn(10, 10, (x, y) => [x * 20, y * 20, 90, 100 + x]);
    expect(Array.from(denoise(img, 0).data)).toEqual(Array.from(img.data));
    const out = denoise(img, 0.5);
    for (let p = 3; p < out.data.length; p += 4) expect(out.data[p]).toBe(img.data[p]);
  });
});

describe('sharpen and clarity', () => {
  it('increases edge energy on a soft image', () => {
    const soft = fromFn(64, 64, (x) => {
      const v = 60 + 140 / (1 + Math.exp(-(x - 32) / 2.5));
      return [v, v, v];
    });
    const lap = (im) => laplacianVariance(toLuma(im), im.width, im.height);
    const sharp = sharpen(soft, { amount: 1.5, sigma: 1.2, threshold: 0 });
    expect(lap(sharp)).toBeGreaterThan(lap(soft) * 1.5);
    expect(lap(clarity(soft, 1))).toBeGreaterThan(lap(soft));
  });

  it('ignores noise below the threshold', () => {
    const img = addNoise(solid(40, 40, [128, 128, 128, 255]), 0.6);
    const out = sharpen(img, { amount: 2, sigma: 1, threshold: 6 });
    expect(Array.from(out.data)).toEqual(Array.from(img.data));
  });
});

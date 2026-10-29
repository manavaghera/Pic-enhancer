import { describe, expect, it } from 'vitest';
import { adjustTone, autoTone, neutralMean, whiteBalanceGains } from '../src/core/color.js';
import { fromFn, solid } from './helpers.js';

const range = (img, ch = 0) => {
  let min = 255, max = 0;
  for (let p = ch; p < img.data.length; p += 4) {
    min = Math.min(min, img.data[p]);
    max = Math.max(max, img.data[p]);
  }
  return [min, max];
};

describe('autoTone', () => {
  it('expands a low-contrast image, limited to 2x gain', () => {
    const img = fromFn(100, 10, (x) => {
      const v = 80 + x * 0.9;
      return [v, v, v];
    });
    const [min, max] = range(autoTone(img, 1));
    expect(min).toBeLessThan(50);
    expect(max).toBeGreaterThan(200);
    expect(max - min).toBeGreaterThanOrEqual(170);
  });

  it('stretches a dark image without lifting black', () => {
    const img = fromFn(100, 10, (x) => [x * 1.6, x * 1.6, x * 1.6]);
    const [min, max] = range(autoTone(img, 1));
    expect(min).toBeLessThanOrEqual(1);
    expect(max).toBeGreaterThan(240);
  });

  it('neutralises a colour cast', () => {
    const out = autoTone(solid(20, 20, [146, 122, 122, 255]), 1);
    const [r, g, b] = out.data;
    expect(Math.abs(r - g)).toBeLessThanOrEqual(3);
    expect(Math.abs(g - b)).toBeLessThanOrEqual(3);
  });

  it('does not "correct" a legitimately colourful scene', () => {
    const scene = fromFn(40, 40, (x, y) => (y < 20 ? [80, 130, 210] : [60, 140, 50]));
    expect(whiteBalanceGains(scene)).toEqual([1, 1, 1]);
  });

  it('does nothing at strength 0', () => {
    const img = solid(4, 4, [150, 120, 120, 255]);
    expect(Array.from(autoTone(img, 0).data)).toEqual(Array.from(img.data));
  });
});

describe('adjustTone', () => {
  const img = fromFn(16, 16, (x, y) => [x * 16, y * 16, 128, 200]);

  it('is identity with default values', () => {
    expect(Array.from(adjustTone(img, {}).data)).toEqual(Array.from(img.data));
  });

  it('desaturates fully at saturation -1', () => {
    const out = adjustTone(img, { saturation: -1 });
    for (let p = 0; p < out.data.length; p += 4) {
      expect(Math.abs(out.data[p] - out.data[p + 1])).toBeLessThanOrEqual(1);
      expect(Math.abs(out.data[p + 1] - out.data[p + 2])).toBeLessThanOrEqual(1);
    }
  });

  it('brightens with positive brightness and keeps alpha', () => {
    const gray = solid(4, 4, [100, 100, 100, 77]);
    const out = adjustTone(gray, { brightness: 0.5 });
    expect(out.data[0]).toBeGreaterThan(100);
    expect(out.data[3]).toBe(77);
  });

  it('warmth shifts red up and blue down', () => {
    const out = adjustTone(solid(2, 2, [128, 128, 128, 255]), { warmth: 1 });
    expect(out.data[0]).toBeGreaterThan(128);
    expect(out.data[2]).toBeLessThan(128);
  });
});

describe('neutralMean', () => {
  it('ignores a tint confined to one small tonal band', () => {
    // Mostly saturated sunset colours plus a purple-grey ridge (10% of pixels).
    const img = fromFn(100, 100, (x, y) => (y < 10 ? [114, 97, 113] : y < 55 ? [230, 120, 60] : [60, 50, 140]));
    expect(neutralMean(img)).toBeNull();
  });

  it('accepts a tint that is consistent across shadows and highlights', () => {
    const img = fromFn(100, 100, (x, y) => (y < 50 ? [80, 70, 62] : [200, 182, 166]));
    const m = neutralMean(img);
    expect(m[0]).toBeGreaterThan(m[2]);
  });
});

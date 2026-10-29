import { describe, expect, it } from 'vitest';
import {
  analyzeImage, aspectLabel, blockiness, countColors, edgeWidth, estimateGrain, estimateNoise, laplacianVariance,
} from '../src/core/analyze.js';
import { gaussianBlur } from '../src/core/filters.js';
import { toLuma } from '../src/core/image.js';
import { dominantColors } from '../src/core/palette.js';
import { addNoise, checker, fromFn, rng, solid } from './helpers.js';

const lumaOf = (img) => toLuma(img);

describe('noise estimate', () => {
  it('measures gaussian noise close to its true sigma', () => {
    const img = addNoise(solid(256, 256, [128, 128, 128, 255]), 10);
    const sigma = estimateNoise(lumaOf(img), 256, 256);
    expect(sigma).toBeGreaterThan(7);
    expect(sigma).toBeLessThan(12);
  });

  it('reports almost no noise for a clean graphic', () => {
    expect(estimateNoise(lumaOf(checker(256, 256, 32)), 256, 256)).toBeLessThan(1);
  });
});

describe('edge width', () => {
  const discs = () => {
    const rand = rng(3);
    const shapes = Array.from({ length: 30 }, () => [rand() * 256, rand() * 256, 5 + rand() * 30, 40 + rand() * 180]);
    return fromFn(256, 256, (x, y) => {
      let v = 90 + x * 0.2;
      for (const [cx, cy, r, val] of shapes) if ((x - cx) ** 2 + (y - cy) ** 2 < r * r) v = val;
      return [v, v, v];
    });
  };

  it('grows with blur and stays robust to noise', () => {
    const L = lumaOf(discs());
    const sharp = edgeWidth(L, 256, 256);
    const soft = edgeWidth(gaussianBlur(L, 256, 256, 2), 256, 256);
    const noisySharp = edgeWidth(lumaOf(addNoise(discs(), 8)), 256, 256);
    expect(sharp).toBeLessThan(3.5);
    expect(soft).toBeGreaterThan(7);
    expect(noisySharp).toBeLessThan(4);
  });

  it('returns null without real edges', () => {
    expect(edgeWidth(lumaOf(fromFn(64, 64, (x) => [x, x, x])), 64, 64)).toBeNull();
  });
});

describe('grain', () => {
  it('catches low-frequency grain that the full-scale estimate misses', () => {
    const rand = rng(4);
    const blotches = Array.from({ length: 64 * 64 }, () => (rand() - 0.5) * 30);
    const img = fromFn(256, 256, (x, y) => {
      const v = 128 + blotches[(y >> 2) * 64 + (x >> 2)];
      return [v, v, v];
    });
    const L = lumaOf(gaussianBlurImg(img));
    expect(estimateGrain(L, 256, 256)).toBeGreaterThan(estimateNoise(L, 256, 256) * 1.3);
  });
});

function gaussianBlurImg(img) {
  const L = gaussianBlur(lumaOf(img), img.width, img.height, 1);
  return fromFn(img.width, img.height, (x, y) => {
    const v = L[y * img.width + x];
    return [v, v, v];
  });
}

describe('sharpness', () => {
  it('scores a blurred image lower than the sharp original', () => {
    const img = checker(128, 128, 8);
    const L = lumaOf(img);
    const blurred = gaussianBlur(L, 128, 128, 2.5);
    expect(laplacianVariance(L, 128, 128)).toBeGreaterThan(laplacianVariance(blurred, 128, 128) * 4);
  });
});

describe('blockiness', () => {
  it('flags 8x8 block patterns and ignores smooth images', () => {
    const rand = rng(5);
    const levels = Array.from({ length: 256 }, () => 60 + rand() * 120);
    const blocky = fromFn(128, 128, (x, y) => {
      const v = levels[(y >> 3) * 16 + (x >> 3)] + (x % 8) * 0.5;
      return [v, v, v];
    });
    const smooth = fromFn(128, 128, (x, y) => [x + y / 2, x + y / 2, x + y / 2]);
    expect(blockiness(lumaOf(blocky), 128, 128)).toBeGreaterThan(1.5);
    expect(blockiness(lumaOf(smooth), 128, 128)).toBeLessThan(1.15);
  });
});

describe('palette', () => {
  it('finds the two colours of a split image', () => {
    const img = fromFn(40, 40, (x) => (x < 20 ? [255, 0, 0] : [0, 0, 255]));
    const pal = dominantColors(img, 6);
    expect(pal.length).toBe(2);
    expect(pal.map((p) => p.hex).sort()).toEqual(['#0000ff', '#ff0000']);
    expect(pal[0].pct).toBeCloseTo(0.5, 1);
  });
});

describe('analyzeImage', () => {
  it('describes a grayscale gradient', () => {
    const img = fromFn(300, 200, (x) => [x * 0.85, x * 0.85, x * 0.85]);
    const a = analyzeImage(img);
    expect(a.width).toBe(300);
    expect(a.aspect).toBe('3:2');
    expect(a.isGrayscale).toBe(true);
    expect(a.cast.label).toBe('Neutral');
    expect(a.histogram.l.reduce((s, v) => s + v, 0)).toBe(300 * 200);
    expect(a.score).toBeGreaterThanOrEqual(0);
    expect(a.score).toBeLessThanOrEqual(100);
  });

  it('detects a colour cast and transparency', () => {
    const img = fromFn(64, 64, (x, y) => [130 + x, 112 + x, 96 + x, y < 8 ? 0 : 255]);
    const a = analyzeImage(img);
    expect(a.cast.label).toBe('Warm');
    expect(a.hasAlpha).toBe(true);
    expect(a.transparentPct).toBeCloseTo(8 / 64, 2);
  });

  it('marks tiny low-colour images as pixel art', () => {
    const img = fromFn(32, 32, (x, y) => ((x >> 2) + (y >> 2)) % 3 === 0 ? [255, 200, 0] : [30, 30, 80]);
    const a = analyzeImage(img);
    expect(a.isPixelArt).toBe(true);
    expect(a.uniqueColors).toBe(2);
  });

  it('tells clean flat-filled art from photos', () => {
    // A cartoon: flat fills with dark outlines, too big to be pixel art.
    const cartoon = fromFn(640, 400, (x, y) => {
      const d = Math.hypot(x - 300, y - 200);
      if (Math.abs(d - 120) < 3 || Math.abs(y - 330) < 2) return [25, 20, 30];
      if (d < 120) return [250, 205, 170];
      return y > 330 ? [90, 170, 70] : [120, 190, 240];
    });
    const art = analyzeImage(cartoon);
    expect(art.isPixelArt).toBe(false);
    expect(art.isIllustration).toBe(true);
    // The same picture with camera grain is a photo.
    expect(analyzeImage(addNoise(cartoon, 4)).isIllustration).toBe(false);
  });

  it('handles 1x1 images', () => {
    const a = analyzeImage(solid(1, 1, [10, 20, 30, 255]));
    expect(a.width).toBe(1);
    expect(Number.isFinite(a.score)).toBe(true);
  });
});

describe('helpers', () => {
  it('labels common aspect ratios', () => {
    expect(aspectLabel(1920, 1080)).toBe('16:9');
    expect(aspectLabel(1080, 1350)).toBe('4:5');
    expect(aspectLabel(1000, 540)).toBe('1.85:1');
  });

  it('caps colour counting', () => {
    const img = fromFn(100, 100, (x, y) => [x, y, 0]);
    expect(countColors(img, 4096)).toBe(4097);
    expect(countColors(solid(5, 5, [1, 2, 3, 255]))).toBe(1);
  });
});

describe('analyzeImage with a reference size', () => {
  it('measures detail metrics on the reference image', () => {
    const clean = checker(128, 128, 32);
    const noisy = addNoise(clean, 12);
    const a = analyzeImage(noisy, { detailImage: clean });
    expect(a.width).toBe(128);
    expect(a.noise.sigma).toBeLessThan(1);
  });
});

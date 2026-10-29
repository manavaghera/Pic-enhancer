import { describe, expect, it } from 'vitest';
import { analyzeImage } from '../src/core/analyze.js';
import { gaussianBlur } from '../src/core/filters.js';
import { AI_MODELS } from '../src/ai/models.js';
import {
  AI_MAX_INPUT_PIXELS,
  aiInputLimit,
  aiInputSize,
  computeTargetSize,
  DEFAULT_SETTINGS,
  enhance,
  MAX_OUTPUT_PIXELS,
  PRESETS,
  tuneForAI,
  upscaleSteps,
} from '../src/core/pipeline.js';
import { resize } from '../src/core/resize.js';
import { describePlan, recommend } from '../src/core/recommend.js';
import { settingsFor, state } from '../src/state.js';
import { addNoise, channelStats, fromFn, solid } from './helpers.js';

const scene = (w, h) =>
  fromFn(w, h, (x, y) => {
    const sky = 120 + (y / h) * 80;
    const disc = (x - w * 0.3) ** 2 + (y - h * 0.4) ** 2 < (h * 0.2) ** 2;
    if (y > h * 0.9) return [18, 20, 24];
    return disc ? [230, 190, 60] : y > h * 0.65 ? [60, 110 + (x % 7), 50] : [sky * 0.6, sky * 0.8, sky];
  });

describe('computeTargetSize', () => {
  it('handles multipliers and resolution targets', () => {
    expect(computeTargetSize(1000, 500, '2x')).toMatchObject({ width: 2000, height: 1000, capped: false });
    expect(computeTargetSize(1000, 500, '4k')).toMatchObject({ width: 3840, height: 1920 });
    expect(computeTargetSize(500, 1000, '4k')).toMatchObject({ width: 1920, height: 3840 });
    expect(computeTargetSize(5000, 3000, '4k')).toMatchObject({ width: 5000, height: 3000, factor: 1 });
    expect(computeTargetSize(800, 600, '1x')).toMatchObject({ width: 800, height: 600 });
  });

  it('caps the output at the 8K UHD pixel count', () => {
    const t = computeTargetSize(4000, 3000, '8k');
    expect(t.capped).toBe(true);
    expect(t.width * t.height).toBeLessThanOrEqual(MAX_OUTPUT_PIXELS);
    expect(t.width / t.height).toBeCloseTo(4 / 3, 2);
    const huge = computeTargetSize(9000, 6000, '2x');
    expect(huge).toMatchObject({ width: 9000, height: 6000, factor: 1, capped: true });
  });

  it('uses whole-number factors for pixel art', () => {
    const t = computeTargetSize(100, 60, '4k', { integer: true });
    expect(t.factor).toBe(38);
    expect(t.width).toBe(3800);
  });

  it('rejects unknown scales', () => {
    expect(() => computeTargetSize(10, 10, '3x')).toThrow();
  });
});

describe('upscaleSteps', () => {
  it('doubles progressively in detailed mode', () => {
    const t = computeTargetSize(100, 50, '4x');
    expect(upscaleSteps(100, 50, t, false)).toEqual([[400, 200]]);
    expect(upscaleSteps(100, 50, t, true)).toEqual([[200, 100], [400, 200]]);
    expect(upscaleSteps(100, 50, computeTargetSize(100, 50, '1x'), true)).toEqual([]);
  });
});

describe('enhance', () => {
  it('returns the input unchanged when every step is off', async () => {
    const img = scene(40, 30);
    const off = { ...DEFAULT_SETTINGS, scale: '1x', denoise: 0, sharpen: 0, clarity: 0, autoColor: false };
    const { image } = await enhance(img, off);
    expect(image).not.toBe(img);
    expect(Array.from(image.data)).toEqual(Array.from(img.data));
  });

  it('runs every preset at several scales without changing overall brightness much', async () => {
    const img = addNoise(scene(60, 40), 6);
    const meanIn = channelStats(img, 1).mean;
    for (const [id, preset] of Object.entries(PRESETS)) {
      for (const [scale, mode] of [['2x', 'standard'], ['4x', 'detailed']]) {
        const { image, target } = await enhance(img, { ...preset.values, scale, mode });
        expect(image.width, id).toBe(target.width);
        expect(image.height, id).toBe(target.height);
        expect(image.width, id).toBe(scale === '2x' ? 120 : 240);
        expect(Math.abs(channelStats(image, 1).mean - meanIn), id).toBeLessThan(40);
      }
    }
  });

  it('reports monotonic progress ending at 1', async () => {
    const seen = [];
    await enhance(scene(50, 50), { ...DEFAULT_SETTINGS, scale: '4x', mode: 'detailed' }, (f) => seen.push(f));
    expect(seen.at(-1)).toBe(1);
    for (let i = 1; i < seen.length; i++) expect(seen[i]).toBeGreaterThanOrEqual(seen[i - 1] - 1e-9);
  });
});

// Stand-in for the neural network: plain 4x nearest-neighbour.
const fakeAI = async (img, model, onProgress) => {
  onProgress?.(1);
  return resize(img, img.width * 4, img.height * 4, { method: 'nearest' });
};

describe('enhance with the AI engine', () => {
  const ai = { ...DEFAULT_SETTINGS, engine: 'ai', aiModel: 'photo', denoise: 0, sharpen: 0, clarity: 0, autoColor: false };

  it('fits the 4x model output to every scale option', async () => {
    const img = scene(50, 30);
    for (const [scale, w] of [['1x', 50], ['2x', 100], ['4x', 200], ['4k', 3840]]) {
      const { image } = await enhance(img, { ...ai, scale }, () => {}, { aiUpscale: fakeAI });
      expect(image.width, scale).toBe(w);
      expect(image.height, scale).toBe(Math.round((30 * w) / 50));
    }
  });

  it('passes the chosen model and reports AI progress', async () => {
    const calls = [];
    const labels = new Set();
    const spy = async (img, model, p) => {
      calls.push([img.width, model]);
      return fakeAI(img, model, p);
    };
    await enhance(scene(40, 40), { ...ai, aiModel: 'anime', scale: '4x' }, (f, label) => labels.add(label), { aiUpscale: spy });
    expect(calls).toEqual([[40, 'anime']]);
    expect([...labels].some((l) => l.startsWith('AI upscaling'))).toBe(true);
  });

  it('scales oversized inputs down so the AI result fits the pixel cap', async () => {
    let aiInput = 0;
    const spy = async (img, model, p) => {
      aiInput = img.width * img.height;
      return fakeAI(img, model, p);
    };
    const big = solid(1800, 1300, [90, 120, 150, 255]);
    const { image } = await enhance(big, { ...ai, scale: '2x' }, () => {}, { aiUpscale: spy });
    expect(aiInput).toBeLessThanOrEqual(AI_MAX_INPUT_PIXELS);
    expect(image.width * image.height).toBeLessThanOrEqual(MAX_OUTPUT_PIXELS);
    expect(image.width).toBe(3600);
  }, 60_000);

  it('keeps the heavy HQ model within its own, smaller input limit', async () => {
    let aiInput = 0;
    const spy = async (img, model, p) => {
      aiInput = img.width * img.height;
      return fakeAI(img, model, p);
    };
    const limit = AI_MODELS.hq.maxInputPixels;
    expect(aiInputLimit('hq')).toBe(limit);
    expect(aiInputLimit('photo')).toBe(AI_MAX_INPUT_PIXELS);
    const big = solid(1300, 1000, [90, 120, 150, 255]);
    const { image } = await enhance(big, { ...ai, aiModel: 'hq', scale: '2x' }, () => {}, { aiUpscale: spy });
    expect(aiInput).toBeLessThanOrEqual(limit);
    expect(aiInput).toBeGreaterThan(limit * 0.99);
    expect([image.width, image.height]).toEqual([2600, 2000]);
  }, 60_000);

  it('works on a shrunk copy of blurry pictures and still fills the output size', async () => {
    let aiInput = null;
    const spy = async (img, model, p) => {
      aiInput = [img.width, img.height];
      return fakeAI(img, model, p);
    };
    expect(aiInputSize(400, 300, { aiModel: 'hq', aiPrescale: 0.5 })).toEqual([200, 150]);
    expect(aiInputSize(400, 300, { aiModel: 'hq' })).toEqual([400, 300]);
    // The fast models turn shrunk copies crunchy, so they keep the full size.
    expect(aiInputSize(400, 300, { aiModel: 'photo', aiPrescale: 0.5 })).toEqual([400, 300]);
    const { image } = await enhance(scene(400, 300), { ...ai, aiModel: 'hq', aiPrescale: 0.5, scale: '2x' }, () => {}, { aiUpscale: spy });
    expect(aiInput).toEqual([200, 150]);
    expect([image.width, image.height]).toEqual([800, 600]);
  });

  it('adds generative detail from the local AI at the output size', async () => {
    const calls = [];
    const refine = async (img, strength) => {
      calls.push([img.width, img.height, strength]);
      return { ...img, data: new Uint8ClampedArray(img.data).fill(200) };
    };
    const labels = new Set();
    const { image } = await enhance(scene(600, 400), { ...ai, scale: '4x', refine: 0.35, sharpen: 50 }, (f, l) => labels.add(l), { aiUpscale: fakeAI, refine });
    expect(calls).toEqual([[2400, 1600, 0.35]]);
    expect([image.width, image.height]).toEqual([2400, 1600]);
    expect(image.data[0]).toBe(200);
    expect(image.data[3]).toBe(255);
    // Generated detail only gets its own light crisping, not the usual pass.
    expect([...labels]).toContain('Adding realistic detail (local AI)');
    expect([...labels]).not.toContain('Sharpening details');
  });

  it('gives the local AI the full output, up to 4096 px', async () => {
    const sizes = [];
    const refine = async (img) => (sizes.push([img.width, img.height]), img);
    const classic = { ...DEFAULT_SETTINGS, denoise: 0, sharpen: 0, clarity: 0, autoColor: false, refine: 0.3 };
    await enhance(scene(1200, 800), { ...classic, scale: '4k' }, () => {}, { refine });
    await enhance(scene(1200, 800), { ...classic, scale: '8k' }, () => {}, { refine });
    expect(sizes).toEqual([[3840, 2560], [4096, 2731]]);
  }, 60_000);

  it('keeps the upscaled image when the local AI skips, and needs a runtime', async () => {
    const plain = await enhance(scene(40, 30), { ...ai, scale: '2x' }, () => {}, { aiUpscale: fakeAI });
    const skipped = await enhance(scene(40, 30), { ...ai, scale: '2x', refine: 0.35 }, () => {}, { aiUpscale: fakeAI, refine: async () => null });
    expect(Array.from(skipped.image.data)).toEqual(Array.from(plain.image.data));
    await expect(enhance(scene(40, 30), { ...ai, refine: 0.35 }, () => {}, { aiUpscale: fakeAI })).rejects.toThrow(/Generative detail/);
  });

  it('pins the downloaded HQ model', () => {
    const { url, sha256, bytes } = AI_MODELS.hq;
    expect(url).toMatch(/^https:\/\/huggingface\.co\/.+\.onnx$/);
    expect(sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(bytes).toBeGreaterThan(1e6);
  });

  it('keeps transparency and does not leak hidden colour into edges', async () => {
    const img = fromFn(32, 32, (x) => (x < 16 ? [0, 0, 255, 255] : [255, 0, 0, 0]));
    const { image } = await enhance(img, { ...ai, scale: '4x' }, () => {}, { aiUpscale: fakeAI });
    let transparent = 0;
    for (let p = 0; p < image.data.length; p += 4) {
      if (image.data[p + 3] === 0) transparent++;
      else expect(image.data[p]).toBeLessThan(20);
    }
    expect(transparent).toBeGreaterThan(0);
  });

  it('uses classic nearest-neighbour scaling for pixel art even in AI mode', async () => {
    let used = false;
    const spy = async (...args) => {
      used = true;
      return fakeAI(...args);
    };
    await enhance(scene(16, 16), { ...ai, ...PRESETS.pixel.values, scale: '4x' }, () => {}, { aiUpscale: spy });
    expect(used).toBe(false);
  });

  it('fails clearly without an AI runtime or with an unknown model', async () => {
    await expect(enhance(scene(8, 8), { ...ai, scale: '2x' })).rejects.toThrow(/not available/);
    await expect(enhance(scene(8, 8), { ...ai, aiModel: 'nope' }, () => {}, { aiUpscale: fakeAI })).rejects.toThrow(/Unknown AI model/);
  });
});

describe('tuneForAI', () => {
  it('turns classic denoise and sharpening down for the AI engine', () => {
    expect(tuneForAI({ ...DEFAULT_SETTINGS, denoise: 40, sharpen: 50 })).toMatchObject({ denoise: 0, sharpen: 15 });
    expect(tuneForAI({ ...DEFAULT_SETTINGS, denoise: 65 }).denoise).toBe(25);
    expect(tuneForAI({ ...DEFAULT_SETTINGS, contrast: 15 }).contrast).toBe(15);
    expect(tuneForAI({ ...DEFAULT_SETTINGS, contrast: 15, clarity: 20, vibrance: 25 }, true))
      .toMatchObject({ autoColor: false, contrast: 0, clarity: 0, vibrance: 25 });
  });
});

describe('recommend', () => {
  it('suggests upscaling, sharpening and denoising for a small, soft, noisy photo', () => {
    const base = scene(240, 160);
    const d = base.data;
    for (let c = 0; c < 3; c++) {
      const ch = new Float32Array(240 * 160);
      for (let i = 0; i < ch.length; i++) ch[i] = d[i * 4 + c];
      const blurred = gaussianBlur(ch, 240, 160, 3);
      for (let i = 0; i < ch.length; i++) d[i * 4 + c] = blurred[i];
    }
    const img = addNoise(base, 12, 9, false);
    const { settings, issues } = recommend(analyzeImage(img));
    expect(['4k', '4x']).toContain(settings.scale);
    expect(settings.mode).toBe('detailed');
    expect(settings.denoise).toBeGreaterThanOrEqual(40);
    expect(settings.sharpen).toBeGreaterThanOrEqual(30);
    expect(issues.some((i) => i.title === 'Low resolution')).toBe(true);
  });

  it('picks nearest-neighbour scaling for pixel art', () => {
    const sprite = fromFn(24, 24, (x, y) => ((x >> 3) + (y >> 3)) % 2 ? [250, 80, 80] : [20, 20, 40]);
    const { settings } = recommend(analyzeImage(sprite));
    expect(settings.resample).toBe('nearest');
    expect(settings.sharpen).toBe(0);
  });

  it('picks the upscaler and face restoration for what the device can run', () => {
    const photo = analyzeImage(addNoise(scene(240, 160), 6));
    const gpu = { webgpu: true, webgl: true };
    expect(recommend(photo, {}, gpu).settings).toMatchObject({
      engine: 'ai', aiModel: 'hq', faces: 100, faceModel: 'codeformer', autoColor: false, clarity: 0,
    });
    expect(recommend(photo, {}, { webgpu: false, webgl: true }).settings).toMatchObject({ engine: 'ai', aiModel: 'photo' });
    expect(recommend(photo).settings).toMatchObject({ engine: 'classic', faces: 100 });
    // The heavy model only takes inputs it would not have to shrink much.
    expect(recommend({ ...photo, width: 1500, height: 1000 }, {}, gpu).settings.aiModel).toBe('photo');
    expect(recommend({ ...photo, width: 2400, height: 1600 }, {}, gpu).settings.engine).toBe('classic');
    // Drawings get the art model and keep their own faces.
    expect(recommend({ ...photo, isIllustration: true }, {}, gpu).settings).toMatchObject({ engine: 'ai', aiModel: 'anime', faces: 0 });
  });

  it('shrinks blurry, blown-up photos to their real detail level for the AI', () => {
    const photo = analyzeImage(addNoise(scene(240, 160), 6));
    const gpu = { webgpu: true, webgl: true };
    const blurry = (w, h, edgeWidth) => ({ ...photo, width: w, height: h, sharpness: { ...photo.sharpness, edgeWidth, nativeEdgeWidth: edgeWidth } });
    // A 988 px picture whose edges are ~7 px wide holds ~440 px of detail.
    expect(recommend(blurry(988, 716, 6.8), {}, gpu).settings.aiPrescale).toBeCloseTo(0.441, 3);
    // Never below 400 px on the long side.
    expect(recommend(blurry(600, 400, 9), {}, gpu).settings.aiPrescale).toBeCloseTo(400 / 600, 3);
    expect(recommend(blurry(300, 200, 9), {}, gpu).settings.aiPrescale).toBe(1);
    // Sharp and merely soft pictures keep every pixel; so does a big phone
    // photo that is a little soft at full size (it stays on classic scaling).
    expect(recommend(blurry(988, 716, 4), {}, gpu).settings.aiPrescale).toBe(1);
    expect(recommend(blurry(988, 716, 5.8), {}, gpu).settings.aiPrescale).toBe(1);
    expect(recommend(blurry(4000, 3000, 5), {}, gpu).settings).toMatchObject({ engine: 'classic', aiPrescale: 1 });
    // Never below 40%, however blurry.
    expect(recommend(blurry(1500, 1000, 20), {}, gpu).settings.aiPrescale).toBe(0.4);
    // A big blurry picture is small enough for AI once shrunk.
    expect(recommend(blurry(2000, 1500, 7), {}, gpu).settings).toMatchObject({ engine: 'ai', aiModel: 'hq' });
    // Without the HQ model it stays too big for the fast one.
    expect(recommend(blurry(2000, 1500, 7), {}, { webgpu: false, webgl: true }).settings.engine).toBe('classic');
    expect(describePlan(recommend(blurry(988, 716, 6.8), {}, gpu).settings)[0].detail).toMatch(/44% copy/);
  });

  it('keeps pixel art and documents away from AI and face restoration', () => {
    const gpu = { webgpu: true, webgl: true };
    const sprite = analyzeImage(fromFn(24, 24, (x, y) => ((x >> 3) + (y >> 3)) % 2 ? [250, 80, 80] : [20, 20, 40]));
    expect(recommend(sprite, {}, gpu).settings).toMatchObject({ engine: 'classic', faces: 0, resample: 'nearest' });
    const doc = analyzeImage(fromFn(300, 200, (x, y) => (y % 20 < 3 && x % 9 < 6 ? [30, 30, 30] : [245, 245, 240])));
    expect(doc.isDocument).toBe(true);
    expect(recommend(doc, {}, gpu).settings).toMatchObject({ engine: 'classic', faces: 0 });
  });

  it('describes its plan in plain words', () => {
    const plan = describePlan({
      ...DEFAULT_SETTINGS, engine: 'ai', aiModel: 'hq', faces: 100, faceModel: 'codeformer', denoise: 25, brightness: 20, vibrance: 10,
    });
    expect(plan.map((s) => s.title)).toEqual(['AI upscaling · Real-ESRGAN x4plus', 'Face restoration', 'Noise cleanup', 'Colour fixes', 'Sharpening']);
    expect(plan[3].detail).toBe('Brighter shadows, richer colours, balanced white.');
    // Steps that need a downloaded model say which one.
    expect(plan.map((s) => s.model)).toEqual(['hq', 'codeformer', undefined, undefined, undefined]);
    const pixel = describePlan({ ...DEFAULT_SETTINGS, ...PRESETS.pixel.values });
    expect(pixel.map((s) => s.title)).toEqual(['Pixel-perfect scaling']);
  });

  it('sends every AI step to the local AI server while it runs', () => {
    const photo = recommend(analyzeImage(addNoise(scene(240, 160), 6)), {}, { webgpu: false, webgl: true });
    const art = recommend({ ...analyzeImage(addNoise(scene(240, 160), 6)), isIllustration: true }, {}, { webgpu: true, webgl: true });
    const sprite = recommend(analyzeImage(fromFn(24, 24, (x, y) => ((x >> 3) + (y >> 3)) % 2 ? [250, 80, 80] : [20, 20, 40])));
    state.localAI = true;
    try {
      expect(settingsFor({ width: 240, height: 160, recommendation: photo })).toMatchObject({
        remote: true, engine: 'ai', aiModel: 'hq', faceModel: 'codeformer', refine: 0.45,
      });
      // Drawings use the server's HQ upscaler and keep their own faces and look.
      expect(settingsFor({ width: 240, height: 160, recommendation: art })).toMatchObject({ remote: true, aiModel: 'hq', faces: 0, refine: 0 });
      expect(settingsFor({ width: 24, height: 24, recommendation: sprite }).remote).toBe(false);
      const plan = describePlan(settingsFor({ width: 240, height: 160, recommendation: photo }));
      expect(plan[0].detail).toMatch(/local AI server/);
      // Nothing for the browser to download.
      expect(plan.every((step) => step.model === undefined)).toBe(true);
    } finally {
      state.localAI = false;
    }
  });

  it('applies only the chosen output quality on top of the per-image plan', () => {
    const recommendation = recommend(analyzeImage(addNoise(scene(240, 160), 6)), {}, { webgpu: true, webgl: true });
    const item = { width: 240, height: 160, recommendation };
    state.quality = 'auto';
    expect(settingsFor(item)).toBe(recommendation.settings);
    state.quality = '2x';
    expect(settingsFor(item)).toMatchObject({ ...recommendation.settings, scale: '2x', mode: 'standard' });
    state.quality = '8k';
    expect(settingsFor(item)).toMatchObject({ engine: 'ai', scale: '8k', mode: 'detailed' });
    state.quality = 'auto';
  });

  it('warns about GPS metadata', () => {
    const { issues } = recommend(analyzeImage(scene(50, 50)), { exif: { hasGps: true } });
    expect(issues.some((i) => i.title.includes('GPS'))).toBe(true);
  });
});

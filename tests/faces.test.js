import { describe, expect, it } from 'vitest';
import {
  applyTransform, cropLooksSoft, cropToTensor, FFHQ_512, invertTransform, pasteFromTensor, similarityTransform,
} from '../src/ai/faceGeometry.js';
import { animalShare, CLASSIFIER_SIZE, classifierInput } from '../src/ai/animal.js';
import { decodeYunet, nms, YUNET_SIZE, yunetInput } from '../src/ai/yunet.js';
import { DEFAULT_SETTINGS, enhance, PRESETS } from '../src/core/pipeline.js';
import { resize } from '../src/core/resize.js';
import { fromFn, rng } from './helpers.js';

describe('similarity transform', () => {
  it('recovers a known rotation, scale and shift', () => {
    const truth = [0.8 * Math.cos(0.3), 0.8 * Math.sin(0.3), 40, -12];
    const src = [[10, 20], [80, 25], [45, 60], [20, 90], [75, 88]];
    const dst = src.map(([x, y]) => applyTransform(truth, x, y));
    const m = similarityTransform(src, dst);
    m.forEach((v, i) => expect(v).toBeCloseTo(truth[i], 6));
  });

  it('inverts exactly', () => {
    const m = similarityTransform(FFHQ_512.map(([x, y]) => [x * 0.3 + 100, y * 0.3 + 50]), FFHQ_512);
    const inv = invertTransform(m);
    const [u, v] = applyTransform(m, 123, 77);
    const [x, y] = applyTransform(inv, u, v);
    expect(x).toBeCloseTo(123, 6);
    expect(y).toBeCloseTo(77, 6);
  });
});

describe('crop and paste', () => {
  const rand = rng(8);
  const img = fromFn(300, 240, (x, y) => [x % 256, y % 256, (x * y) % 256].map((v) => v * 0.6 + rand() * 40));
  // A face whose landmarks are the template shrunk to ~38% around (150, 120).
  const landmarks = FFHQ_512.map(([x, y]) => [150 + (x - 256) * 0.38, 120 + (y - 256) * 0.38]);
  const m = similarityTransform(landmarks, FFHQ_512);

  it('produces a normalised 3x512x512 tensor', () => {
    const t = cropToTensor(img, m, 512);
    expect(t.length).toBe(3 * 512 * 512);
    for (let i = 0; i < t.length; i += 4099) {
      expect(t[i]).toBeGreaterThanOrEqual(-1);
      expect(t[i]).toBeLessThanOrEqual(1);
    }
  });

  it('round-trips: pasting the unchanged crop leaves a smooth image almost identical', () => {
    const smooth = fromFn(300, 240, (x, y) => [x * 0.8, y, (x + y) * 0.4]);
    const out = { ...smooth, data: new Uint8ClampedArray(smooth.data) };
    pasteFromTensor(out, m, cropToTensor(smooth, m, 512), 512, 1);
    let maxDiff = 0;
    for (let i = 0; i < out.data.length; i++) maxDiff = Math.max(maxDiff, Math.abs(out.data[i] - smooth.data[i]));
    expect(maxDiff).toBeLessThanOrEqual(2);
  });

  it('only touches the head region, with full effect at the centre', () => {
    const smooth = fromFn(300, 240, () => [120, 120, 120]);
    // A restored face with stripes but the same average colour.
    const t = new Float32Array(3 * 512 * 512);
    for (let i = 0; i < t.length; i++) t[i] = (Math.floor((i % 512) / 32) % 2 ? 0.3 : -0.3) + (120 / 127.5 - 1);
    const out = { ...smooth, data: new Uint8ClampedArray(smooth.data) };
    pasteFromTensor(out, m, t, 512, 1);
    const at = (x, y) => out.data[(y * 300 + x) * 4];
    const centre = [];
    for (let x = 140; x < 160; x++) centre.push(at(x, 128));
    expect(Math.max(...centre) - Math.min(...centre)).toBeGreaterThan(60);
    expect(at(10, 10)).toBe(120);
    expect(at(290, 230)).toBe(120);
  });

  it('matches the restored face to the surrounding colour', () => {
    const smooth = fromFn(300, 240, (x, y) => [140 + x * 0.05, 110, 90 + y * 0.05]);
    const crop = cropToTensor(smooth, m, 512);
    // The model "returned" a face that is much bluer than the photo around it.
    for (let i = 2 * 512 * 512; i < crop.length; i++) crop[i] += 0.4;
    const out = { ...smooth, data: new Uint8ClampedArray(smooth.data) };
    pasteFromTensor(out, m, crop, 512, 1);
    const p = (128 * 300 + 150) * 4;
    expect(Math.abs(out.data[p + 2] - smooth.data[p + 2])).toBeLessThanOrEqual(3);
  });
});

describe('soft face check', () => {
  // A crude face: dark eyes, nose and mouth on skin, with a blurred background.
  const face = fromFn(512, 512, (x, y) => {
    if ((x - 256) ** 2 / 190 ** 2 + (y - 290) ** 2 / 240 ** 2 > 1) return [120 + ((x >> 5) % 2) * 10, 130, 140];
    if (Math.hypot(x - 193, y - 240) < 18 || Math.hypot(x - 319, y - 240) < 18) return [40, 30, 30];
    if (Math.abs(x - 256) < 6 && y > 250 && y < 320) return [150, 100, 90];
    if (Math.abs(y - 371) < 6 && x > 200 && x < 312) return [120, 40, 50];
    return [225, 180, 160];
  });
  const identity = [1, 0, 0, 0];

  it('passes a sharp face and flags the same face blurred', () => {
    expect(cropLooksSoft(cropToTensor(face, identity, 512), 512)).toBe(false);
    const blurred = resize(resize(face, 64, 64), 512, 512);
    expect(cropLooksSoft(cropToTensor(blurred, identity, 512), 512)).toBe(true);
  });
});

describe('animal check', () => {
  it('feeds the classifier a normalised square around the face', () => {
    // Left half red, right half blue; the box sits on the boundary.
    const img = fromFn(200, 100, (x) => (x < 100 ? [255, 0, 0] : [0, 0, 255]));
    const t = classifierInput(img, [80, 40, 40, 20]);
    const plane = CLASSIFIER_SIZE * CLASSIFIER_SIZE;
    expect(t.length).toBe(3 * plane);
    const at = (c, x, y) => t[c * plane + y * CLASSIFIER_SIZE + x];
    expect(at(0, 5, 112)).toBeCloseTo((1 - 0.485) / 0.229, 2);
    expect(at(2, 5, 112)).toBeCloseTo((0 - 0.406) / 0.225, 2);
    expect(at(2, 218, 112)).toBeCloseTo((1 - 0.406) / 0.225, 2);
    // Rows above the picture repeat its top edge instead of going black.
    expect(at(0, 5, 0)).toBeCloseTo(at(0, 5, 112), 2);
  });

  it('measures the belief on animal classes', () => {
    const logits = new Float32Array(1000);
    expect(animalShare(logits)).toBeCloseTo(0.398, 3);
    logits[207] = 12; // golden retriever
    expect(animalShare(logits)).toBeGreaterThan(0.99);
    logits[207] = 0;
    logits[903] = 12; // wig
    expect(animalShare(logits)).toBeLessThan(0.01);
  });
});

describe('YuNet decoding', () => {
  function outputs(cells) {
    const o = {};
    for (const s of [8, 16, 32]) {
      const n = (YUNET_SIZE / s) ** 2;
      o[`cls_${s}`] = new Float32Array(n);
      o[`obj_${s}`] = new Float32Array(n);
      o[`bbox_${s}`] = new Float32Array(n * 4);
      o[`kps_${s}`] = new Float32Array(n * 10);
    }
    for (const { stride, row, col, score } of cells) {
      const idx = row * (YUNET_SIZE / stride) + col;
      o[`cls_${stride}`][idx] = score;
      o[`obj_${stride}`][idx] = score;
      o[`bbox_${stride}`].set([0.5, 0.5, Math.log(4), Math.log(5)], idx * 4);
      o[`kps_${stride}`].set([-1, -1, 1, -1, 0, 0, -1, 1, 1, 1], idx * 10);
    }
    return o;
  }

  it('decodes boxes and landmarks in input pixels', () => {
    const [face] = decodeYunet(outputs([{ stride: 16, row: 10, col: 20, score: 0.9 }]));
    expect(face.score).toBeCloseTo(0.9, 5);
    [(20.5 - 2) * 16, (10.5 - 2.5) * 16, 64, 80].forEach((v, i) => expect(face.box[i]).toBeCloseTo(v, 4));
    expect(face.landmarks[0]).toEqual([19 * 16, 9 * 16]);
    expect(face.landmarks[2]).toEqual([20 * 16, 10 * 16]);
  });

  it('drops weak detections and merges duplicates', () => {
    const faces = decodeYunet(outputs([
      { stride: 16, row: 10, col: 20, score: 0.9 },
      { stride: 16, row: 10, col: 21, score: 0.8 },
      { stride: 32, row: 2, col: 2, score: 0.3 },
    ]));
    expect(faces.length).toBe(1);
    expect(faces[0].score).toBeCloseTo(0.9, 5);
    expect(nms([{ score: 1, box: [0, 0, 10, 10] }, { score: 0.5, box: [50, 50, 10, 10] }], 0.3).length).toBe(2);
  });

  it('letterboxes into a BGR tensor and reports the scale', () => {
    const { data, scale } = yunetInput(fromFn(1280, 640, () => [10, 20, 30]));
    expect(scale).toBeCloseTo(0.5, 5);
    const plane = YUNET_SIZE * YUNET_SIZE;
    expect([data[0], data[plane], data[2 * plane]]).toEqual([30, 20, 10]);
    expect(data[639 * YUNET_SIZE]).toBe(0);
  });
});

describe('enhance with face restoration', () => {
  const base = { ...DEFAULT_SETTINGS, scale: '2x', denoise: 0, sharpen: 0, clarity: 0, autoColor: false };
  const img = fromFn(40, 30, (x, y) => [x * 5, y * 6, 90]);

  it('runs the face step last and reports how many faces it restored', async () => {
    const calls = [];
    const faceRestore = async (im, strength, p, reference, model) => {
      calls.push([im.width, strength, reference.width, reference.data[820], model]);
      return { image: im, faces: 3 };
    };
    const { image, meta } = await enhance(img, { ...base, faces: 70, faceModel: 'codeformer', brightness: 50 }, () => {}, { faceRestore });
    expect(calls[0][4]).toBe('codeformer');
    // The reference is the source-size image after colour/tone steps.
    expect(calls.length).toBe(1);
    expect(calls[0].slice(0, 3)).toEqual([80, 0.7, 40]);
    expect(calls[0][3]).toBeGreaterThan(img.data[820]);
    expect(image.width).toBe(80);
    expect(meta.faces).toBe(3);
  });

  it('runs before the local AI detail, which blends the faces in', async () => {
    const order = [];
    const faceRestore = async (im) => (order.push('faces'), { image: im, faces: 1 });
    const refine = async (im) => (order.push('refine'), im);
    const aiUpscale = async (im) => resize(im, im.width * 4, im.height * 4, { method: 'nearest' });
    await enhance(img, { ...base, engine: 'ai', faces: 100, refine: 0.45 }, () => {}, { faceRestore, refine, aiUpscale });
    expect(order).toEqual(['faces', 'refine']);
  });

  it('is skipped when off or for pixel art, and needs a runtime when on', async () => {
    let used = false;
    const faceRestore = async (im) => {
      used = true;
      return { image: im, faces: 0 };
    };
    await enhance(img, base, () => {}, { faceRestore });
    await enhance(img, { ...base, ...PRESETS.pixel.values, faces: 80 }, () => {}, { faceRestore });
    expect(used).toBe(false);
    await expect(enhance(img, { ...base, faces: 80 })).rejects.toThrow(/not available/);
    await expect(enhance(img, { ...base, faces: 80, faceModel: 'nope' }, () => {}, { faceRestore })).rejects.toThrow(/Unknown face model/);
  });
});

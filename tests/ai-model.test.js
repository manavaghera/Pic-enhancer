// Runs the real bundled Real-ESRGAN models on TensorFlow.js' CPU backend.
// Slow (seconds per 64x64 tile), so inputs are kept to a single tile.

import { readFileSync } from 'node:fs';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { AI_MODELS } from '../src/ai/models.js';
import { looksSane, onnxTileRunner, tileRunner, upscaleWithModel } from '../src/ai/realesrgan.js';
import { remoteOrt, remoteSession } from '../src/ai/remote.js';
import { upscaleTiled } from '../src/ai/tiling.js';
import { edgeWidth } from '../src/core/analyze.js';
import { toLuma } from '../src/core/image.js';
import { resize } from '../src/core/resize.js';
import { fromFn } from './helpers.js';

const PUBLIC = new URL('../public/', import.meta.url);

async function loadModel(id) {
  const base = new URL(AI_MODELS[id].path, PUBLIC);
  const json = JSON.parse(readFileSync(base, 'utf8'));
  const bins = json.weightsManifest.flatMap((g) => g.paths.map((p) => readFileSync(new URL(p, base))));
  const weightData = new Uint8Array(bins.reduce((n, b) => n + b.length, 0));
  let o = 0;
  for (const b of bins) {
    weightData.set(b, o);
    o += b.length;
  }
  return loadGraphModel({
    load: async () => ({
      modelTopology: json.modelTopology,
      format: json.format,
      generatedBy: json.generatedBy,
      convertedBy: json.convertedBy,
      signature: json.signature,
      userDefinedMetadata: json.userDefinedMetadata,
      weightSpecs: json.weightsManifest.flatMap((g) => g.weights),
      weightData: weightData.buffer,
    }),
  });
}

// Crisp shapes, shrunk 4x so the model has real detail to restore.
const truth = fromFn(160, 96, (x, y) => {
  const inDisc = (cx, cy, r) => (x - cx) ** 2 + (y - cy) ** 2 < r * r;
  if (inDisc(40, 48, 26)) return [230, 200, 60];
  if (inDisc(115, 40, 20)) return [40, 60, 160];
  if (y > 70 && x % 24 < 12) return [30, 30, 30];
  return [200, 215, 230];
});
const small = resize(truth, 40, 24);

describe('bundled Real-ESRGAN models', () => {
  const loaded = {};
  beforeAll(async () => {
    await tf.setBackend('cpu');
    for (const id of Object.keys(AI_MODELS)) if (AI_MODELS[id].path) loaded[id] = await loadModel(id);
  }, 60_000);

  it('declare 64px tiles at 4x', () => {
    for (const model of Object.values(loaded)) {
      const { tile, scale } = tileRunner(tf, model);
      expect(tile).toBe(64);
      expect(scale).toBe(4);
    }
  }, 120_000);

  it('pass the backend sanity check used in the browser', async () => {
    expect(await looksSane(tileRunner(tf, loaded.photo))).toBe(true);
  }, 120_000);

  it('upscale faithfully and with sharper edges than Lanczos', async () => {
    const out = await upscaleWithModel(tf, loaded.photo, small);
    expect(out.width).toBe(160);
    expect(out.height).toBe(96);

    const back = resize(out, 40, 24);
    let diff = 0;
    for (let p = 0; p < back.data.length; p += 4) {
      for (let c = 0; c < 3; c++) diff += Math.abs(back.data[p + c] - small.data[p + c]);
    }
    expect(diff / (40 * 24 * 3)).toBeLessThan(12);

    const lanczos = resize(small, 160, 96);
    const width = (im) => edgeWidth(toLuma(im), im.width, im.height);
    expect(width(out)).toBeLessThan(width(lanczos));
  }, 120_000);
});

describe('ONNX tile runner', () => {
  // Stands in for ONNX Runtime: a "model" that upscales NCHW tensors 4x by
  // repeating pixels, so the expected output is a nearest-neighbour resize.
  const ort = {
    Tensor: class {
      constructor(type, data, dims) {
        Object.assign(this, { type, data, dims });
      }
    },
  };
  const nearest4x = {
    inputNames: ['input'],
    outputNames: ['output'],
    async run({ input }) {
      const [, c, h, w] = input.dims;
      const data = new Float32Array(c * h * w * 16);
      for (let k = 0; k < c; k++) {
        for (let y = 0; y < h * 4; y++) {
          for (let x = 0; x < w * 4; x++) data[(k * h * 4 + y) * w * 4 + x] = input.data[(k * h + (y >> 2)) * w + (x >> 2)];
        }
      }
      return { output: new ort.Tensor('float32', data, [1, c, h * 4, w * 4]) };
    },
  };

  it('converts between HWC tiles and NCHW tensors', async () => {
    const img = fromFn(21, 13, (x, y) => [x * 12, y * 19, (x * y) % 256]);
    const runner = onnxTileRunner(ort, nearest4x, 8, 4);
    const out = await upscaleTiled(img, { ...runner, pad: 2 }, runner.run);
    expect(out.data).toEqual(resize(img, 84, 52, { method: 'nearest' }).data);
    expect(await looksSane(onnxTileRunner(ort, nearest4x, 32, 4))).toBe(true);
  });

  it('rejects a backend that returns garbage or the wrong size', async () => {
    const nan = { ...nearest4x, run: async (f) => ({ output: { ...(await nearest4x.run(f)).output, data: new Float32Array(3 * 32 * 32 * 16).fill(NaN) } }) };
    expect(await looksSane(onnxTileRunner(ort, nan, 32, 4))).toBe(false);
    await expect(onnxTileRunner(ort, nearest4x, 32, 2).run(new Float32Array(32 * 32 * 3))).rejects.toThrow(/output size/);
  });
});

describe('models on the local AI server', () => {
  const realFetch = globalThis.fetch;
  afterEach(() => {
    globalThis.fetch = realFetch;
  });

  it('sends raw tensors and reads every output back', async () => {
    const calls = [];
    // A stand-in server: "hq" upscales 4x by repeating pixels; any model also
    // returns a second, tiny output to check that outputs are split apart.
    globalThis.fetch = async (url, { body }) => {
      const u = new URL(url);
      const [, c, h, w] = u.searchParams.get('shape').split(',').map(Number);
      calls.push([u.pathname, u.searchParams.get('weight')]);
      const up = new Float32Array(c * h * w * 16);
      for (let k = 0; k < c; k++) {
        for (let y = 0; y < h * 4; y++) {
          for (let x = 0; x < w * 4; x++) up[(k * h * 4 + y) * w * 4 + x] = body[(k * h + (y >> 2)) * w + (x >> 2)];
        }
      }
      const both = new Float32Array(up.length + 2);
      both.set(up);
      both.set([7, 8], up.length);
      const outputs = [{ name: 'output', shape: [1, c, h * 4, w * 4] }, { name: 'extra', shape: [1, 2] }];
      return new Response(both.buffer, { headers: { 'X-Outputs': JSON.stringify(outputs) } });
    };
    const img = fromFn(21, 13, (x, y) => [x * 12, y * 19, (x * y) % 256]);
    const runner = onnxTileRunner(remoteOrt, remoteSession('hq'), 8, 4);
    const out = await upscaleTiled(img, { ...runner, pad: 2 }, runner.run);
    expect(out.data).toEqual(resize(img, 84, 52, { method: 'nearest' }).data);
    expect(calls[0]).toEqual(['/run/hq', null]);

    const res = await remoteSession('codeformer').run({
      input: new remoteOrt.Tensor('float32', new Float32Array(3 * 4), [1, 3, 2, 2]),
      weight: new remoteOrt.Tensor('float32', new Float32Array([0.5]), []),
    });
    expect(calls.at(-1)).toEqual(['/run/codeformer', '0.5']);
    expect(res.output.dims).toEqual([1, 3, 8, 8]);
    expect(Array.from(res.extra.data)).toEqual([7, 8]);
  });

  it('reports the error message from the server', async () => {
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'Model hq is not on this server.' }), { status: 404 });
    const session = remoteSession('hq');
    await expect(session.run({ input: new remoteOrt.Tensor('float32', new Float32Array(3), [1, 3, 1, 1]) })).rejects.toThrow(/not on this server/);
  });
});

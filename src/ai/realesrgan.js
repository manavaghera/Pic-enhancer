// Real-ESRGAN in the browser. Loaded on demand inside the worker, so Classic
// mode never downloads it. The fast models run on TensorFlow.js (WebGPU, then
// WebGL); the HQ model (x4plus) runs on ONNX Runtime with WebGPU. Each backend's
// output is checked on a test tile before it is trusted.

import { AI_MODELS, AI_SCALE } from './models.js';
import { upscaleTiled } from './tiling.js';

const PAD = 10;

let runtime = null;
const models = new Map();

// Graph models declare their input shape but not their output shape, so the
// scale factor is measured once with a blank tile.
const geometry = new WeakMap();
function modelGeometry(tf, model) {
  if (!geometry.has(model)) {
    const [, tile] = model.inputs[0].shape;
    const out = tf.tidy(() => model.predict(tf.zeros([1, tile, tile, 3])));
    const scale = out.shape[1] / tile;
    out.dispose();
    if (!Number.isInteger(scale) || scale < 1) throw new Error('Unexpected AI model output size.');
    geometry.set(model, { tile, scale });
  }
  return geometry.get(model);
}

// Runs one tile through a loaded graph model.
export function tileRunner(tf, model) {
  const { tile, scale } = modelGeometry(tf, model);
  const run = async (input) => {
    const out = tf.tidy(() => model.predict(tf.tensor4d(input, [1, tile, tile, 3])));
    try {
      return await out.data();
    } finally {
      out.dispose();
    }
  };
  return { tile, scale, run };
}

export function upscaleWithModel(tf, model, img, onProgress) {
  const { tile, scale, run } = tileRunner(tf, model);
  return upscaleTiled(img, { tile, scale, pad: PAD }, run, onProgress);
}

// A smooth gradient should come back almost unchanged; a broken GPU path
// (NaNs, garbage, wrong precision) will not. Takes a tile runner.
export async function looksSane({ tile, scale, run }) {
  const input = new Float32Array(tile * tile * 3);
  for (let y = 0; y < tile; y++) {
    for (let x = 0; x < tile; x++) {
      const q = (y * tile + x) * 3;
      const v = 0.2 + (0.6 * x) / tile;
      input[q] = v;
      input[q + 1] = 0.3 + (0.4 * y) / tile;
      input[q + 2] = 1 - v;
    }
  }
  const out = await run(input);
  const span = tile * scale;
  let err = 0, n = 0;
  for (let y = 8; y < tile - 8; y++) {
    for (let x = 8; x < tile - 8; x++) {
      const o = ((y * scale + 1) * span + x * scale + 1) * 3;
      const q = (y * tile + x) * 3;
      for (let c = 0; c < 3; c++) {
        if (!Number.isFinite(out[o + c])) return false;
        err += Math.abs(out[o + c] - input[q + c]);
        n++;
      }
    }
  }
  return err / n < 0.05;
}

async function loadRuntime() {
  const tf = await import('@tensorflow/tfjs-core');
  const { loadGraphModel } = await import('@tensorflow/tfjs-converter');
  await import('@tensorflow/tfjs-backend-webgl');
  if (self.navigator?.gpu) {
    try {
      await import('@tensorflow/tfjs-backend-webgpu');
    } catch {
      /* WebGPU is optional; WebGL remains */
    }
  }
  return { tf, loadGraphModel, backend: null, failed: new Set() };
}

async function useBackend(tf, failed) {
  for (const name of ['webgpu', 'webgl']) {
    if (failed.has(name) || !tf.findBackendFactory(name)) continue;
    try {
      if (await tf.setBackend(name)) {
        await tf.ready();
        return name;
      }
    } catch {
      /* try the next backend */
    }
    failed.add(name);
  }
  return null;
}

function forgetModels() {
  for (const m of models.values()) m.dispose();
  models.clear();
}

async function ensureModel(id) {
  const spec = AI_MODELS[id];
  if (!spec) throw new Error(`Unknown AI model: ${id}`);
  runtime ??= await loadRuntime();
  const { tf, loadGraphModel, failed } = runtime;
  for (;;) {
    runtime.backend ??= await useBackend(tf, failed);
    if (!runtime.backend) {
      throw new Error('AI upscaling needs WebGPU or WebGL, which this browser does not provide. Use Classic instead.');
    }
    if (models.has(id)) return models.get(id);
    let model;
    try {
      model = await loadGraphModel(new URL(`${import.meta.env.BASE_URL}${spec.path}`, self.location.href).href);
    } catch {
      throw new Error('Could not download the AI model. Check your connection and try again.');
    }
    if (await looksSane(tileRunner(tf, model))) {
      models.set(id, model);
      return model;
    }
    model.dispose();
    failed.add(runtime.backend);
    runtime.backend = null;
    forgetModels();
  }
}

// Adapts an NCHW ONNX session (any input size) to the HWC tiles used above.
export function onnxTileRunner(ort, session, tile, scale) {
  const plane = tile * tile, span = tile * scale, outPlane = span * span;
  const chw = new Float32Array(3 * plane);
  const run = async (input) => {
    for (let i = 0; i < plane; i++) {
      chw[i] = input[i * 3];
      chw[plane + i] = input[i * 3 + 1];
      chw[2 * plane + i] = input[i * 3 + 2];
    }
    const outputs = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, tile, tile]) });
    const out = outputs[session.outputNames[0]];
    if (out.dims[2] !== span || out.dims[3] !== span) throw new Error('Unexpected AI model output size.');
    const d = out.data;
    const hwc = new Float32Array(3 * outPlane);
    for (let i = 0; i < outPlane; i++) {
      hwc[i * 3] = d[i];
      hwc[i * 3 + 1] = d[outPlane + i];
      hwc[i * 3 + 2] = d[2 * outPlane + i];
    }
    out.dispose?.();
    return hwc;
  };
  return { tile, scale, run };
}

// The ONNX model is too heavy for the CPU (many minutes per photo), so it
// needs WebGPU. Only one is kept loaded: { id, session, runner }.
let onnxModel = null;

async function ensureOnnx(id, onProgress, cachedOnly = false) {
  if (onnxModel?.id === id) return onnxModel.runner;
  const spec = AI_MODELS[id];
  const { createSession, downloadVerified, ort } = await import('./onnx.js');
  const bytes = await downloadVerified(spec, (f) => onProgress?.(f, `Downloading ${spec.name} (${spec.size}, once)`), { cachedOnly });
  onProgress?.(1, `Loading ${spec.name}`);
  let session;
  try {
    ({ session } = await createSession(bytes, ['webgpu']));
  } catch {
    throw new Error(`${spec.label} needs WebGPU, which this browser does not provide. Use the Photo model instead.`);
  }
  const runner = onnxTileRunner(ort, session, spec.tile, AI_SCALE);
  let sane = false;
  try {
    sane = await looksSane(runner);
  } catch {
    /* treated as a broken GPU path */
  }
  if (!sane) {
    await session.release();
    throw new Error(`${spec.label} does not run correctly on this GPU. Use the Photo model instead.`);
  }
  await onnxModel?.session.release();
  onnxModel = { id, session, runner };
  return runner;
}

// Frees the GPU memory held by loaded models (they load again when next
// needed), so the local AI server can have the GPU to itself.
export async function releaseAI() {
  await onnxModel?.session.release();
  onnxModel = null;
  forgetModels();
}

// True when a model can run without downloading anything first.
export async function aiModelReady(id) {
  if (!AI_MODELS[id]?.url || onnxModel?.id === id) return true;
  return (await import('./onnx.js')).isCached(AI_MODELS[id]);
}

// Loads (and verifies) a model ahead of time; returns the backend in use.
export async function prepareAI(id, onProgress) {
  if (AI_MODELS[id]?.url) {
    await ensureOnnx(id, onProgress);
    return { backend: 'webgpu' };
  }
  await ensureModel(id);
  return { backend: runtime.backend };
}

// 4x upscale of an RGBA image (alpha is ignored and returned opaque).
// cachedOnly: see downloadVerified. remote: run the model on the local AI
// server (it has the downloaded models), leaving the browser off the GPU.
export async function aiUpscale(img, id, onProgress, { cachedOnly = false, remote = false } = {}) {
  if (remote) {
    if (!AI_MODELS[id]?.url) throw new Error(`${AI_MODELS[id]?.label ?? id} does not run on the local AI server.`);
    const { remoteOrt, remoteSession } = await import('./remote.js');
    const runner = onnxTileRunner(remoteOrt, remoteSession(id), AI_MODELS[id].tile, AI_SCALE);
    return upscaleTiled(img, { ...runner, pad: PAD }, runner.run, onProgress);
  }
  if (AI_MODELS[id]?.url) {
    const runner = await ensureOnnx(id, (f, label) => onProgress?.(0, `${label} ${Math.round(f * 100)}%`), cachedOnly);
    return upscaleTiled(img, { ...runner, pad: PAD }, runner.run, onProgress);
  }
  const model = await ensureModel(id);
  try {
    return await upscaleWithModel(runtime.tf, model, img, onProgress);
  } catch (err) {
    // A WebGPU device can be lost mid-run; retry once on WebGL.
    if (runtime.backend !== 'webgpu') throw err;
    runtime.failed.add('webgpu');
    runtime.backend = null;
    forgetModels();
    return upscaleWithModel(runtime.tf, await ensureModel(id), img, onProgress);
  }
}

// Background worker: decoding, analysis, enhancement and encoding all happen
// here so the page stays responsive. Images never leave the device.

import { FACE_MODELS } from './ai/faceModels.js';
import { AI_MODELS, LOCAL_AI_URL } from './ai/models.js';
import { analyzeImage } from './core/analyze.js';
import { estimateJpegQuality, readExif, sniffFormat } from './core/exif.js';
import { hasTransparency } from './core/image.js';
import { enhance, MAX_INPUT_PIXELS } from './core/pipeline.js';
import { resize } from './core/resize.js';

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function decode(blob, format) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob);
  } catch {
    if (format === 'HEIC') throw new Error('HEIC photos are not supported by this browser. Convert to JPG or PNG first.');
    throw new Error('This file could not be decoded as an image.');
  }
  const { width, height } = bitmap;
  if (width * height > MAX_INPUT_PIXELS) {
    bitmap.close();
    throw new Error(`Image is ${(width * height / 1e6).toFixed(0)} MP. The limit is 50 MP.`);
  }
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return { width, height, data: ctx.getImageData(0, 0, width, height).data };
}

async function encode(img, mime, quality) {
  const canvas = new OffscreenCanvas(img.width, img.height);
  canvas.getContext('2d').putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  let target = canvas;
  if (mime === 'image/jpeg' && hasTransparency(img)) {
    target = new OffscreenCanvas(img.width, img.height);
    const ctx = target.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, img.width, img.height);
    ctx.drawImage(canvas, 0, 0);
  }
  try {
    return await target.convertToBlob({ type: mime, quality });
  } catch {
    throw new Error('The browser could not encode an image this large. Try a smaller output size.');
  }
}

function throttle(id) {
  let last = 0;
  return (fraction, label) => {
    const now = performance.now();
    if (now - last < 60 && fraction < 1) return;
    last = now;
    post({ id, type: 'progress', fraction, label });
  };
}

const handlers = {
  async analyze({ file }) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const format = sniffFormat(bytes);
    const img = await decode(file, format);
    return {
      format,
      exif: readExif(bytes),
      jpegQuality: estimateJpegQuality(bytes),
      analysis: analyzeImage(img),
    };
  },

  // The settings were chosen for this device, but a GPU can still refuse a
  // model, so each AI step falls back (HQ -> Photo -> Classic, faces -> skipped)
  // and says so in notes rather than failing the image. Downloaded models are
  // only used once saved (the loader worker fetches them), so a run never
  // waits on a download; the ones it had to skip are listed in pending.
  // fetchNow: models this browser cannot save, which may be fetched inline.
  async enhance({ file, settings, fetchNow = [] }, progress) {
    progress(0, 'Decoding');
    const started = performance.now();
    const src = await decode(file, null);
    const notes = [];
    const pending = [];
    const cachedOnly = (model) => !fetchNow.includes(model);
    // While the local AI server runs it does all AI work, and nothing falls
    // back to the browser's GPU (sharing the GPU crashed the graphics driver).
    const remote = Boolean(settings.remote);
    const runtimes = {
      async aiUpscale(img, model, onProgress) {
        const ai = await loadAI();
        try {
          return await ai.aiUpscale(img, model, onProgress, { cachedOnly: cachedOnly(model), remote });
        } catch (err) {
          if (model === 'photo' || model === 'anime' || remote) throw err;
          if (err.code === 'NOT_CACHED') pending.push(model);
          else notes.push(`${AI_MODELS[model].label} could not run here (${err.message}), so the Photo model was used.`);
          return ai.aiUpscale(img, 'photo', onProgress);
        }
      },
      // The local AI server (server/app.py); its tile count drives progress.
      // The browser's models are freed first: running both on one GPU at
      // once can exhaust its memory and crash the graphics driver.
      async refine(img, strength, onProgress) {
        await releaseGPU();
        const poll = setInterval(() => {
          fetch(`${LOCAL_AI_URL}/progress`).then((r) => r.json()).then((p) => p.total && onProgress(p.done / p.total)).catch(() => {});
        }, 700);
        try {
          const res = await fetch(`${LOCAL_AI_URL}/refine?strength=${strength}`, {
            method: 'POST', headers: { 'Content-Type': 'image/png' }, body: await encode(img, 'image/png'),
          });
          if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail || `HTTP ${res.status}`);
          return await decode(await res.blob(), null);
        } catch (err) {
          notes.push(`Realistic detail was skipped: the local AI server did not finish (${err.message}).`);
          return null;
        } finally {
          clearInterval(poll);
        }
      },
      async faceRestore(img, strength, onProgress, reference, model) {
        try {
          const faces = await loadFaces();
          return await faces.restoreFaces(img, strength, onProgress, reference, model, { cachedOnly: cachedOnly(model), remote });
        } catch (err) {
          if (err.code === 'NOT_CACHED') pending.push(model);
          else notes.push(`Faces were not restored: ${err.message}`);
          return { image: img, faces: null };
        }
      },
    };
    const onProgress = (f, label) => progress(0.02 + f * 0.83, label);
    // A model still downloading is swapped for the fast one up front, so the
    // picture is sized for the model that actually runs.
    if (!remote && settings.engine === 'ai' && cachedOnly(settings.aiModel) && !(await (await loadAI()).aiModelReady(settings.aiModel))) {
      pending.push(settings.aiModel);
      settings = { ...settings, aiModel: 'photo' };
    }
    let out;
    try {
      out = await enhance(src, settings, onProgress, runtimes);
    } catch (err) {
      if (settings.engine !== 'ai') throw err;
      notes.push(`AI upscaling could not run here (${err.message}), so classic upscaling was used.`);
      out = await enhance(src, { ...settings, engine: 'classic' }, onProgress, runtimes);
    }
    const { image, target, meta } = out;
    progress(0.86, 'Analyzing result');
    const sameSize = image.width === src.width && image.height === src.height;
    const analysis = analyzeImage(image, { detailImage: sameSize ? image : resize(image, src.width, src.height) });
    progress(0.9, 'Encoding PNG');
    const blob = await encode(image, 'image/png');
    progress(1, 'Done');
    return {
      blob, width: image.width, height: image.height, capped: target.capped, faces: meta.faces ?? null, analysis,
      notes, pending, ms: performance.now() - started,
    };
  },

  async convert({ blob, mime, quality }) {
    const img = await decode(blob, null);
    return { blob: await encode(img, mime, quality) };
  },

  // Downloads and verifies an AI model ahead of the first run.
  async aiPrepare({ model }, progress) {
    return (await loadAI()).prepareAI(model, progress);
  },

  // Background download of a large model into Cache Storage, in its own
  // worker so runs never wait on it. Reports whether the browser kept it.
  async fetchModel({ model }, progress) {
    const spec = AI_MODELS[model]?.url ? AI_MODELS[model] : FACE_MODELS[model];
    if (!spec?.url) throw new Error(`Unknown model: ${model}`);
    const { downloadVerified, isCached } = await import('./ai/onnx.js');
    await downloadVerified(spec, (f) => progress(f, 'Downloading'));
    return { cached: await isCached(spec) };
  },
};

const releaseGPU = async () => {
  if (aiModule) await (await aiModule).releaseAI();
  if (faceModule) await (await faceModule).releaseFaces();
};

// ONNX Runtime, the face detector and the face model only load when needed.
let faceModule = null;
const loadFaces = () =>
  (faceModule ??= import('./ai/faceRestore.js').catch((err) => {
    faceModule = null;
    throw new Error(`Could not load face restoration (${err.message}).`);
  }));

// TensorFlow.js and the model only load when the AI engine is used.
let aiModule = null;
const loadAI = () =>
  (aiModule ??= import('./ai/realesrgan.js').catch((err) => {
    aiModule = null;
    throw new Error(`Could not load the AI engine (${err.message}).`);
  }));

async function run({ id, type, payload } = {}) {
  const handler = handlers[type];
  if (!handler) {
    post({ id, type: 'error', message: `Unknown task: ${type}` });
    return;
  }
  try {
    const result = await handler(payload, throttle(id));
    post({ id, type: 'result', result });
  } catch (err) {
    post({ id, type: 'error', message: err?.message || 'Something went wrong while processing the image.' });
  }
}

// One job at a time: async decoding would otherwise let a large batch hold
// every decoded image in memory at once.
let queue = Promise.resolve();
self.onmessage = (e) => {
  queue = queue.then(() => run(e.data));
};

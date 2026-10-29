// The enhancement pipeline: denoise -> colour -> tone -> clarity -> upscale
// -> sharpen. Cheap global steps run at source resolution; only resampling
// and final sharpening touch the (possibly 8K) output. The upscale step is
// either classic resampling or the Real-ESRGAN model (engine: 'ai').

import { FACE_MODELS } from '../ai/faceModels.js';
import { AI_MODELS, AI_SCALE, LOCAL_AI_MAX_SIDE } from '../ai/models.js';
import { adjustTone, autoTone } from './color.js';
import { clarity, denoise, sharpen } from './filters.js';
import { bleedTransparent, hasTransparency } from './image.js';
import { resize } from './resize.js';

export const MAX_OUTPUT_PIXELS = 7680 * 4320;
export const MAX_INPUT_PIXELS = 50e6;
// Larger AI inputs are scaled down first so the 4x result stays within the cap
// (and within the model's own limit, for the heavy HQ model).
export const AI_MAX_INPUT_PIXELS = Math.floor(MAX_OUTPUT_PIXELS / (AI_SCALE * AI_SCALE));
export const aiInputLimit = (model) => Math.min(AI_MAX_INPUT_PIXELS, AI_MODELS[model]?.maxInputPixels ?? Infinity);

// Size of the copy the AI model works on: within the model's limit, and
// shrunk by aiPrescale for blurry pictures (see recommend.js) if the model
// takes that.
export function aiInputSize(w, h, s) {
  const pre = AI_MODELS[s.aiModel]?.prescale ? s.aiPrescale ?? 1 : 1;
  const fit = Math.min(1, Math.sqrt(aiInputLimit(s.aiModel) / (w * h)), pre);
  return [Math.max(1, Math.floor(w * fit)), Math.max(1, Math.floor(h * fit))];
}
export const SCALES = ['1x', '2x', '4x', '4k', '8k'];

export const DEFAULT_SETTINGS = Object.freeze({
  engine: 'classic',
  aiModel: 'photo',
  // Fraction of the source size the AI works on (1 = full size).
  aiPrescale: 1,
  // Generative detail from the local AI server, 0 = off (see models.js).
  refine: 0,
  // Run every AI step on the local AI server instead of the browser.
  remote: false,
  faces: 0,
  faceModel: 'gfpgan',
  scale: '2x',
  mode: 'standard',
  resample: 'lanczos',
  denoise: 20,
  sharpen: 40,
  clarity: 15,
  autoColor: true,
  brightness: 0,
  contrast: 0,
  saturation: 0,
  vibrance: 0,
  warmth: 0,
});

const look = (o) => ({ resample: 'lanczos', brightness: 0, contrast: 0, saturation: 0, vibrance: 0, warmth: 0, ...o });

// Presets change the "look" only; scale and quality mode stay the user's choice.
// AI already removes noise and restores edges, so Auto and the presets dial
// the classic denoise and sharpen steps right down when the AI engine is on.
// natural (used for Auto) also keeps the original colours and contrast, as
// generative enhancers like Artguru do; brightness/vibrance fixes remain.
export function tuneForAI(s, natural = false) {
  const tuned = { ...s, denoise: s.denoise >= 60 ? 25 : 0, sharpen: Math.round(s.sharpen * 0.3) };
  return natural ? { ...tuned, autoColor: false, contrast: 0, clarity: 0 } : tuned;
}

export const PRESETS = {
  photo: { label: 'Photo', values: look({ denoise: 20, sharpen: 45, clarity: 20, autoColor: true, vibrance: 10 }) },
  portrait: { label: 'Portrait', values: look({ denoise: 35, sharpen: 25, clarity: 5, autoColor: true, warmth: 5, vibrance: 5 }) },
  landscape: { label: 'Landscape', values: look({ denoise: 15, sharpen: 50, clarity: 35, autoColor: true, contrast: 10, vibrance: 25 }) },
  lowlight: { label: 'Low light', values: look({ denoise: 65, sharpen: 30, clarity: 15, autoColor: true, brightness: 25, vibrance: 10 }) },
  text: { label: 'Text & scans', values: look({ denoise: 10, sharpen: 80, clarity: 25, autoColor: true, contrast: 25 }) },
  old: { label: 'Old photo', values: look({ denoise: 50, sharpen: 40, clarity: 20, autoColor: true, contrast: 10, vibrance: 10 }) },
  pixel: { label: 'Pixel art', values: look({ resample: 'nearest', denoise: 0, sharpen: 0, clarity: 0, autoColor: false }) },
};

// Output size for a scale option. Never shrinks; caps area at 8K UHD pixels.
export function computeTargetSize(w, h, scale, { integer = false } = {}) {
  const long = Math.max(w, h);
  let factor = { '1x': 1, '2x': 2, '4x': 4 }[scale];
  if (factor === undefined) {
    if (scale !== '4k' && scale !== '8k') throw new Error(`Unknown scale: ${scale}`);
    factor = Math.max(1, (scale === '8k' ? 7680 : 3840) / long);
  }
  let capped = false;
  const maxFactor = Math.sqrt(MAX_OUTPUT_PIXELS / (w * h));
  if (factor > maxFactor) {
    factor = Math.max(1, maxFactor);
    capped = true;
  }
  if (integer) factor = Math.max(1, Math.floor(factor));
  const round = capped ? Math.floor : Math.round;
  return {
    width: Math.max(1, round(w * factor)),
    height: Math.max(1, round(h * factor)),
    factor,
    capped,
  };
}

// Returns [width, height] for each resampling pass. Detailed mode doubles in
// steps (with light sharpening between them) before the exact final size.
export function upscaleSteps(w, h, target, detailed) {
  if (target.width === w && target.height === h) return [];
  const steps = [];
  if (detailed) {
    for (let k = 2; k < target.factor; k *= 2) steps.push([Math.round(w * k), Math.round(h * k)]);
  }
  steps.push([target.width, target.height]);
  return steps;
}

async function aiStep(im, model, aiUpscale, onProgress) {
  const alpha = hasTransparency(im);
  const up = await aiUpscale(alpha ? bleedTransparent(im) : im, model, onProgress);
  if (alpha) {
    const a = resize(im, up.width, up.height).data;
    for (let p = 3; p < up.data.length; p += 4) up.data[p] = a[p];
  }
  return up;
}

function aiStages(src, s, target, aiUpscale) {
  if (!AI_MODELS[s.aiModel]) throw new Error(`Unknown AI model: ${s.aiModel}`);
  if (!aiUpscale) throw new Error('AI upscaling is not available here.');
  const stages = [];
  const n0 = src.width * src.height;
  const [aw, ah] = aiInputSize(src.width, src.height, s);
  if (aw !== src.width || ah !== src.height) {
    stages.push({ label: 'Preparing for AI', cost: n0, run: (im, p) => resize(im, aw, ah, { onProgress: p }) });
  }
  stages.push({
    label: `AI upscaling · ${AI_MODELS[s.aiModel].name}`,
    cost: aw * ah * 300,
    run: (im, p) => aiStep(im, s.aiModel, aiUpscale, p),
  });
  if (aw * AI_SCALE !== target.width || ah * AI_SCALE !== target.height) {
    stages.push({
      label: 'Fitting to output size',
      cost: Math.max(aw * ah * AI_SCALE * AI_SCALE, target.width * target.height) * 1.5,
      run: (im, p) => resize(im, target.width, target.height, { onProgress: p }),
    });
  }
  return stages;
}

// Generative detail at the image's own size, or on a copy LOCAL_AI_MAX_SIDE
// long fitted back for larger images, then lightly crisped (tuned by eye on
// cars and buildings; stronger starts to look crunchy). Transparency is kept
// from the input.
async function refineStep(im, strength, refine, onProgress) {
  const k = Math.min(1, LOCAL_AI_MAX_SIDE / Math.max(im.width, im.height));
  const work = k < 1 ? resize(im, Math.round(im.width * k), Math.round(im.height * k)) : im;
  let out = await refine(work, strength, onProgress);
  if (!out) return im;
  if (out.width !== im.width || out.height !== im.height) out = resize(out, im.width, im.height);
  out = sharpen(out, { amount: 0.6, sigma: 0.8, threshold: 3 });
  for (let p = 3; p < out.data.length; p += 4) out.data[p] = im.data[p];
  return out;
}

function plan(src, s, { aiUpscale, faceRestore, refine }, meta) {
  const pixel = s.resample === 'nearest';
  const ai = s.engine === 'ai' && !pixel;
  const target = computeTargetSize(src.width, src.height, s.scale, { integer: pixel });
  const n0 = src.width * src.height;
  const stages = [];
  const tone = {
    brightness: s.brightness / 100,
    contrast: s.contrast / 100,
    warmth: s.warmth / 100,
    saturation: s.saturation / 100,
    vibrance: s.vibrance / 100,
  };

  // "pre" stages run at source size; their output is the reference image the
  // face step crops from, as GFPGAN expects the original, not an upscale.
  if (s.denoise > 0) stages.push({ pre: true, label: 'Reducing noise', cost: n0 * 4, run: (im, p) => denoise(im, s.denoise / 100, p) });
  if (s.autoColor) stages.push({ pre: true, label: 'Balancing color', cost: n0 * 0.3, run: (im) => autoTone(im, 1) });
  if (Object.values(tone).some((v) => v !== 0)) {
    stages.push({ pre: true, label: 'Adjusting tone', cost: n0 * 0.5, run: (im) => adjustTone(im, tone) });
  }
  if (s.clarity > 0) stages.push({ pre: true, label: 'Adding clarity', cost: n0 * 2, run: (im, p) => clarity(im, s.clarity / 100, p) });

  const detailed = !pixel && !ai && s.mode === 'detailed';
  // Grain left after denoising must not be re-amplified, so the sharpening
  // threshold rises with the noise-reduction strength.
  const threshold = 2 + 8 * (s.denoise / 100);
  if (ai) stages.push(...aiStages(src, s, target, aiUpscale));
  const steps = ai ? [] : upscaleSteps(src.width, src.height, target, detailed);
  steps.forEach(([w, h], i) => {
    const last = i === steps.length - 1;
    stages.push({
      label: pixel ? 'Scaling pixels' : `Upscaling${steps.length > 1 ? ` (pass ${i + 1}/${steps.length})` : ''}`,
      cost: w * h * 1.5,
      run: (im, p) => resize(im, w, h, { method: pixel ? 'nearest' : 'lanczos', onProgress: p }),
    });
    if (detailed && !last) {
      stages.push({
        label: 'Refining edges',
        cost: w * h,
        run: (im, p) => sharpen(im, { amount: 0.35 + 0.4 * (s.sharpen / 100), sigma: 1, threshold, onProgress: p }),
      });
    }
  });

  // Generated detail is already crisp; sharpening it would add halos.
  const refining = s.refine > 0 && !pixel;
  if (s.sharpen > 0 && !pixel && !refining) {
    const f = Math.min(target.factor, 8);
    const amount = (s.sharpen / 100) * (ai ? 1.2 : detailed ? 1.8 : 1.6);
    const sigma = ai ? 1 : Math.min(2.5, 0.7 + 0.25 * f);
    stages.push({
      label: 'Sharpening details',
      cost: target.width * target.height * 1.2,
      run: (im, p) => sharpen(im, { amount, sigma, threshold, onProgress: p }),
    });
  }

  // Generative face restoration comes after every other step, so nothing
  // re-processes the faces, except the local AI's generated detail, which
  // blends the restored faces into the rest of the picture.
  if (s.faces > 0 && !pixel) {
    if (!faceRestore) throw new Error('Face restoration is not available here.');
    if (!FACE_MODELS[s.faceModel]) throw new Error(`Unknown face model: ${s.faceModel}`);
    stages.push({
      label: 'Restoring faces',
      cost: target.width * target.height * 2 + 4e6,
      run: async (im, p, ctx) => {
        const r = await faceRestore(im, s.faces / 100, p, ctx.reference, s.faceModel);
        meta.faces = r.faces;
        return r.image;
      },
    });
  }

  if (refining) {
    if (!refine) throw new Error('Generative detail is not available here.');
    stages.push({
      label: 'Adding realistic detail (local AI)',
      cost: target.width * target.height * 40,
      run: (im, p) => refineStep(im, s.refine, refine, p),
    });
  }
  return { target, stages };
}

// onProgress(fraction 0..1, label); a step may report its own, more specific
// label (a model download, say). Runtimes are injected so the pipeline stays
// pure: aiUpscale(img, modelId, onProgress) -> 4x image (engine 'ai');
// faceRestore(img, strength, onProgress, reference, modelId) -> { image, faces }
// where reference is the same picture at source size (only when faces > 0);
// refine(img, strength, onProgress) -> same-size image, or null to skip the
// step (only when refine > 0).
// Returns { image, target, meta } where meta.faces counts restored faces.
export async function enhance(src, settings, onProgress = () => {}, runtimes = {}) {
  const s = { ...DEFAULT_SETTINGS, ...settings };
  if (src.width * src.height > MAX_INPUT_PIXELS) throw new Error('Image is larger than 50 megapixels.');
  const meta = {};
  const { target, stages } = plan(src, s, runtimes, meta);
  const total = stages.reduce((sum, st) => sum + st.cost, 0) || 1;
  let done = 0;
  let img = src;
  const ctx = { reference: src };
  for (const stage of stages) {
    onProgress(done / total, stage.label);
    img = await stage.run(img, (f, label) => onProgress((done + stage.cost * f) / total, label || stage.label), ctx);
    if (stage.pre) ctx.reference = img;
    done += stage.cost;
  }
  if (img === src) img = { width: src.width, height: src.height, data: new Uint8ClampedArray(src.data) };
  onProgress(1, 'Done');
  return { image: img, target, meta };
}

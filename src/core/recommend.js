// Turns an analysis into human-readable findings and every enhancement
// decision: upscaler, face restoration, cleanup and colour. Pure function:
// same analysis and browser capabilities, same answer.

import { FACE_MODELS } from '../ai/faceModels.js';
import { AI_MODELS } from '../ai/models.js';
import { aiInputLimit, computeTargetSize, DEFAULT_SETTINGS, tuneForAI } from './pipeline.js';

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// What this browser can run. webgpu: a WebGPU adapter with 16-bit float
// shaders (needed by the HQ model); webgl: enough for the fast AI models.
export const NO_GPU = Object.freeze({ webgpu: false, webgl: false });

export function recommend(a, meta = {}, caps = NO_GPU) {
  const issues = [];
  const add = (level, title, detail) => issues.push({ level, title, detail });
  const s = { ...DEFAULT_SETTINGS, autoColor: false, denoise: 0, sharpen: 30, clarity: 10 };
  const long = Math.max(a.width, a.height);

  if (a.isPixelArt) {
    const scale = long <= 128 ? '4k' : long * 4 <= 7680 ? '4x' : '2x';
    Object.assign(s, { resample: 'nearest', scale, denoise: 0, sharpen: 0, clarity: 0 });
    add('info', 'Pixel art or flat graphic', `${a.uniqueColors} colours. Crisp nearest-neighbour scaling keeps hard pixel edges.`);
  } else if (long >= 3840) {
    s.scale = '1x';
    add('good', 'High resolution', `${a.width} × ${a.height} (${a.resolution.tier}) is already print-ready.`);
  } else {
    s.scale = 3840 / long <= 4 ? '4k' : '4x';
    const level = long < 1280 ? 'bad' : 'warn';
    add(level, 'Low resolution', `${a.width} × ${a.height} looks soft on large screens. Upscaling to ${s.scale.toUpperCase()} is recommended.`);
  }
  const factor = computeTargetSize(a.width, a.height, s.scale).factor;
  s.mode = factor > 2 && !a.isPixelArt ? 'detailed' : 'standard';

  if (!a.isPixelArt) {
    if (a.sharpness.edgeWidth == null) {
      s.sharpen = factor > 1 ? 40 : 25;
      add('info', 'Few distinct edges', 'Sharpness could not be measured reliably; using gentle sharpening.');
    } else if (a.sharpness.score < 40) {
      s.sharpen = 75;
      s.clarity = 20;
      add('bad', 'Blurry', 'Edges are soft. Strong sharpening will recover detail.');
    } else if (a.sharpness.score < 70) {
      s.sharpen = 55;
      add('warn', 'Slightly soft', 'Moderate sharpening will make edges crisper.');
    } else {
      s.sharpen = factor > 1 ? 40 : 25;
      add('good', 'Sharp', 'Fine detail is well defined.');
    }

    const sigma = a.noise.sigma;
    if (sigma > 8) {
      s.denoise = 65;
      add('bad', 'Heavy noise', `Grain level σ≈${sigma.toFixed(1)}. Strong noise reduction recommended.`);
    } else if (sigma > 4) {
      s.denoise = 40;
      add('warn', 'Visible noise', `Grain level σ≈${sigma.toFixed(1)}. Light noise reduction will clean it up.`);
    } else {
      s.denoise = sigma > 2.5 ? 15 : 0;
      add('good', 'Clean', `Very little noise (σ≈${sigma.toFixed(1)}).`);
    }

    if (a.blockiness.ratio > 1.3) {
      s.denoise = Math.max(s.denoise, 45);
      add(a.blockiness.ratio > 1.6 ? 'bad' : 'warn', 'Compression artifacts', 'Blocky JPEG patterns detected. Smoothing will reduce them.');
    }
  }

  if (a.exposure.label === 'Underexposed') {
    s.brightness = clamp(Math.round((118 - a.brightness) / 2), 10, 40);
    s.autoColor = true;
    add('warn', 'Underexposed', 'The image is dark. Brightening shadows will reveal detail.');
  } else if (a.exposure.label === 'Overexposed') {
    s.brightness = -clamp(Math.round((a.brightness - 150) / 3), 5, 25);
    s.autoColor = true;
    add('warn', 'Overexposed', 'Highlights are very bright or clipped.');
  }

  if (!a.isDocument && !a.isPixelArt && a.contrast < 40) {
    s.autoColor = true;
    s.contrast = 15;
    add('warn', 'Low contrast', 'The image looks flat. Contrast will be expanded.');
  }

  if (a.cast.label !== 'Neutral' && !a.isGrayscale && !a.isPixelArt) {
    s.autoColor = true;
    add('warn', `${a.cast.label} colour cast`, 'White balance will be neutralised.');
  }

  if (!a.isGrayscale && !a.isDocument && !a.isPixelArt && a.colorfulness < 25) {
    s.vibrance = 25;
    add('info', 'Muted colours', 'A vibrance boost will add life without oversaturating skin tones.');
  }

  if (a.isDocument) {
    // Text edges are far stronger than paper grain, so denoising keeps them
    // intact and stops the strong sharpening from amplifying the grain.
    Object.assign(s, { sharpen: Math.max(s.sharpen, 70), contrast: 20, vibrance: 0, autoColor: true });
    s.denoise = Math.max(s.denoise, 30);
    add('info', 'Document or screenshot', 'Tuned for crisp, readable text.');
  }

  if (a.isIllustration) add('info', 'Illustration or graphic', 'Clean lines and flat colours: the art upscaler keeps them crisp.');

  // A blown-up picture, whose edges are blurred over many pixels even at full
  // size, holds far less detail than its pixel count, and the HQ model, which
  // expects about one pixel per detail, paints its soft blobs into smears. Such
  // pictures go to the AI shrunk to roughly their real detail level (edges ~3
  // px wide), never below 40% or 400 px. Photos that are merely soft keep
  // every pixel: shrinking them would throw real detail away.
  const native = a.sharpness.nativeEdgeWidth ?? a.sharpness.edgeWidth;
  const prescale = native > 6 ? Math.min(1, Math.max(3 / native, 0.4, 400 / long)) : 1;
  // AI upscaling for everything except pixel art and text. Inputs bigger than
  // the model's limit would be shrunk further, losing real detail, so large
  // sharp images keep classic resampling.
  const pixels = a.width * a.height;
  if (!a.isPixelArt && !a.isDocument && (caps.webgpu || caps.webgl)) {
    const hq = caps.webgpu && pixels * prescale * prescale <= aiInputLimit('hq');
    const aiModel = a.isIllustration ? 'anime' : hq ? 'hq' : 'photo';
    if (aiModel === 'hq' || pixels <= aiInputLimit(aiModel)) {
      Object.assign(s, tuneForAI(s, true), { engine: 'ai', aiModel, aiPrescale: Math.round(prescale * 1000) / 1000 });
    }
  }
  // Faces in photos are restored when found; drawings keep their own style.
  if (!a.isPixelArt && !a.isDocument && !a.isIllustration) Object.assign(s, { faces: 100, faceModel: 'codeformer' });

  if (a.hasAlpha) add('info', 'Transparent background', 'Transparency is kept when you download as PNG or WebP.');
  if (meta.exif?.hasGps) add('info', 'Contains GPS location', 'Enhanced downloads do not include metadata, so the location is removed.');

  return { settings: s, issues };
}

// Plain-language steps for the settings, shown as the AI's plan. A step that
// needs a downloaded model names it in model.
export function describePlan(s) {
  const steps = [];
  const step = (title, detail, model) => steps.push({ title, detail, model });
  if (s.resample === 'nearest') step('Pixel-perfect scaling', 'Keeps every hard pixel edge.');
  else if (s.engine === 'ai') {
    const m = AI_MODELS[s.aiModel];
    const shrink = m.prescale && s.aiPrescale < 1 ? ` The photo is blurrier than its size, so AI works on a ${Math.round(s.aiPrescale * 100)}% copy to avoid smears.` : '';
    const where = s.remote ? ' Runs on the local AI server.' : '';
    step(`AI upscaling · ${m.name}`, m.blurb + shrink + where, m.url && !s.remote ? s.aiModel : undefined);
  } else step('Detail-preserving upscaling', 'High-quality resampling that keeps every original pixel.');
  if (s.faces > 0) step('Face restoration', `Small or blurry faces are redrawn with ${FACE_MODELS[s.faceModel].name}.`, s.remote ? undefined : s.faceModel);
  if (s.refine > 0) step('Realistic detail · Stable Diffusion', 'Your GPU redraws realistic fine detail and texture (local AI server).');
  if (s.denoise > 0) step('Noise cleanup', 'Removes grain and JPEG blocks.');
  const color = [
    s.brightness > 0 && 'brighter shadows',
    s.brightness < 0 && 'softer highlights',
    s.contrast > 0 && 'more contrast',
    s.vibrance > 0 && 'richer colours',
    s.autoColor && 'balanced white',
  ].filter(Boolean);
  if (color.length) step('Colour fixes', `${color.join(', ').replace(/^./, (c) => c.toUpperCase())}.`);
  if (s.sharpen > 0 || s.clarity > 0 || s.refine > 0) step('Sharpening', 'Crisper edges and fine detail.');
  return steps;
}

// App state with a tiny subscribe/notify store. Renderers read the whole
// state and update only what changed.

import { LOCAL_AI_STRENGTH, LOCAL_AI_STRENGTH_LARGE, LOCAL_AI_URL } from './ai/models.js';
import { computeTargetSize, DEFAULT_SETTINGS } from './core/pipeline.js';

export const state = {
  items: [],
  selectedId: null,
  // The only choice left to the user: output size, or 'auto' to let each
  // image's analysis pick it. Everything else is decided per image.
  quality: 'auto',
  // Large models fetched in the background, by id:
  // { status: 'checking' | 'downloading' | 'ready' | 'failed', progress, cached }.
  downloads: {},
  // Whether the local AI server (server/app.py) answered its last check.
  localAI: false,
  tab: 'enhance',
  output: { format: 'png', quality: 90 },
  batchRunning: false,
};

const listeners = new Set();
let queued = false;

export function subscribe(fn) {
  listeners.add(fn);
}

// Coalesces bursts of updates (e.g. progress ticks) into one render per frame.
// The timer fallback keeps rendering while rAF is paused in hidden windows.
export function notify() {
  if (queued) return;
  queued = true;
  const run = () => {
    if (!queued) return;
    queued = false;
    for (const fn of listeners) fn(state);
  };
  requestAnimationFrame(run);
  setTimeout(run, 120);
}

export const selectedItem = () => state.items.find((it) => it.id === state.selectedId) || null;
export const findItem = (id) => state.items.find((it) => it.id === id) || null;

// The settings an item would be enhanced with right now: its own analysis'
// decisions, at the chosen output size. While the local AI server runs, every
// AI step goes there (remote), so the browser never uses the GPU at the same
// time: Chrome's WebGPU and the server's CUDA sharing it crashed the graphics
// driver. The server has the HQ upscaler (used for drawings too) and
// CodeFormer, and photos also get generative detail.
export function settingsFor(item) {
  let s = item?.recommendation?.settings || DEFAULT_SETTINGS;
  if (state.localAI && s.resample !== 'nearest') {
    s = { ...s, remote: true, aiModel: s.engine === 'ai' ? 'hq' : s.aiModel };
    // (faces > 0 marks a photo: drawings, text and pixel art keep their look.)
    if (s.faces > 0) s.refine = s.engine === 'ai' ? LOCAL_AI_STRENGTH : LOCAL_AI_STRENGTH_LARGE;
    if (s.faces > 0) s.faceModel = 'codeformer';
  }
  if (state.quality === 'auto' || !item?.width) return s;
  const { factor } = computeTargetSize(item.width, item.height, state.quality);
  return { ...s, scale: state.quality, mode: factor > 2 && s.resample !== 'nearest' ? 'detailed' : 'standard' };
}

export const sameSettings = (a, b) => !!a && !!b && Object.keys(DEFAULT_SETTINGS).every((k) => a[k] === b[k]);

// Checks the local AI server; quick to fail when it is not running. It only
// counts when it has every model the app sends it.
export async function checkLocalAI() {
  let online = false;
  try {
    const health = await (await fetch(`${LOCAL_AI_URL}/health`, { signal: AbortSignal.timeout(1500) })).json();
    online = ['codeformer', 'hq', 'yunet'].every((m) => health.onnx?.includes(m));
  } catch {
    /* not running */
  }
  if (online !== state.localAI) {
    state.localAI = online;
    notify();
  }
  return online;
}

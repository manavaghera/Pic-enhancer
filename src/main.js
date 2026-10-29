import './styles.css';
import { FACE_MODELS } from './ai/faceModels.js';
import { AI_MODELS } from './ai/models.js';
import { NO_GPU, recommend } from './core/recommend.js';
import { zipParts } from './core/zip.js';
import { checkLocalAI, findItem, notify, sameSettings, selectedItem, settingsFor, state, subscribe } from './state.js';
import { createAnalysisView } from './ui/analysisView.js';
import { createControls } from './ui/controls.js';
import { $, download, h, storageGet, storageSet } from './ui/dom.js';
import { icons } from './ui/icons.js';
import { createQueue } from './ui/queue.js';
import { SAMPLES } from './ui/samples.js';
import { toast } from './ui/toast.js';
import { createViewer } from './ui/viewer.js';
import { WorkerClient } from './ui/workerClient.js';

const MAX_FILE_BYTES = 200 * 1024 * 1024;
const MAX_ITEMS = 200;
const IMAGE_EXT = /\.(jpe?g|jfif|png|webp|avif|gif|bmp|heic|heif|tiff?|ico)$/i;
const EXT_BY_TYPE = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp' };

const makeWorker = () => new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const analyzer = new WorkerClient(makeWorker);
const enhancer = new WorkerClient(makeWorker);
// Downloads large models on its own, so enhancing never waits for them.
const loader = new WorkerClient(makeWorker);

let nextId = 1;
let batchCancelled = false;
let viewerKey = null;

// What this browser can run decides which upscaler the analysis picks.
async function detectCaps() {
  let webgl = false;
  try {
    const canvas = document.createElement('canvas');
    webgl = Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'));
  } catch {
    /* no WebGL */
  }
  let webgpu = false;
  try {
    // Some drivers never answer; treat a slow reply as no WebGPU.
    const adapter = await Promise.race([navigator.gpu?.requestAdapter(), new Promise((r) => setTimeout(r, 3000))]);
    webgpu = Boolean(adapter?.features.has('shader-f16'));
  } catch {
    /* no WebGPU */
  }
  return { webgpu, webgl };
}
const capsReady = detectCaps().catch(() => NO_GPU);
checkLocalAI();
window.addEventListener('focus', checkLocalAI);

// ---------- static chrome ----------
$('#brandLogo').innerHTML = icons.logo;
$('#dzIcon').innerHTML = icons.upload;
$('#privacyPill').prepend(h('span', { html: icons.lock }));
$('#zoomIn').innerHTML = icons.zoomIn;
$('#zoomOut').innerHTML = icons.zoomOut;
$('#zoomFit').innerHTML = icons.fit;

const themeBtn = $('#themeToggle');
function setTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  const dark = theme ? theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  themeBtn.innerHTML = dark ? icons.sun : icons.moon;
  themeBtn.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
}
setTheme(storageGet('pixellift-theme'));
themeBtn.addEventListener('click', () => {
  const dark = themeBtn.getAttribute('aria-label') === 'Switch to light mode';
  const next = dark ? 'light' : 'dark';
  storageSet('pixellift-theme', next);
  setTheme(next);
});

// ---------- components ----------
const zoomLabel = $('#zoomLabel');
const viewer = createViewer($('#viewer'), { onZoomChange: (z) => { zoomLabel.textContent = `${Math.round(z * 100)}%`; } });
$('#zoomIn').addEventListener('click', () => viewer.zoomIn());
$('#zoomOut').addEventListener('click', () => viewer.zoomOut());
$('#zoomFit').addEventListener('click', () => viewer.fit());
$('#zoomActual').addEventListener('click', () => viewer.actualSize());

const controls = createControls($('#panelEnhance'), {
  onQuality: (value) => {
    state.quality = value;
    notify();
  },
  onEnhance: async () => {
    const item = selectedItem();
    if (!item?.info || state.batchRunning || state.items.some((it) => it.status === 'processing')) return;
    await checkLocalAI();
    enhanceItem(item, settingsFor(item));
  },
  onCancel: cancelWork,
  onDownload: downloadSelected,
  onOutput: (key, value) => {
    state.output = { ...state.output, [key]: value };
    notify();
  },
});

const analysisView = createAnalysisView($('#panelAnalyze'));

const queue = createQueue($('#queue'), {
  onSelect: select,
  onRemove: removeItem,
  onAdd: () => $('#fileInput').click(),
  onEnhanceAll: enhanceAll,
  onDownloadAll: downloadAll,
  onClear: clearAll,
});

function selectTab(tab) {
  state.tab = tab;
  for (const [id, name] of [['tabEnhance', 'enhance'], ['tabAnalyze', 'analyze']]) {
    $(`#${id}`).setAttribute('aria-selected', String(tab === name));
  }
  $('#panelEnhance').hidden = tab !== 'enhance';
  $('#panelAnalyze').hidden = tab !== 'analyze';
  if (tab === 'analyze') requestAnimationFrame(() => analysisView.refresh());
}
$('#tabEnhance').addEventListener('click', () => selectTab('enhance'));
$('#tabAnalyze').addEventListener('click', () => selectTab('analyze'));
$('.tabs').addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  const next = state.tab === 'enhance' ? 'analyze' : 'enhance';
  selectTab(next);
  $(next === 'enhance' ? '#tabEnhance' : '#tabAnalyze').focus();
});

// ---------- intake ----------
function addFiles(list) {
  let added = 0;
  for (const file of list) {
    const name = file.name || 'pasted-image.png';
    if (/\.svg$/i.test(name) || file.type === 'image/svg+xml') {
      toast(`${name}: SVG is a vector format and is already resolution-independent.`, 'warn');
      continue;
    }
    if (!file.type.startsWith('image/') && !IMAGE_EXT.test(name)) {
      toast(`${name} is not a supported image.`, 'bad');
      continue;
    }
    if (file.size > MAX_FILE_BYTES) {
      toast(`${name} is larger than 200 MB.`, 'bad');
      continue;
    }
    if (state.items.length >= MAX_ITEMS) {
      toast(`You can work on up to ${MAX_ITEMS} images at a time.`, 'warn');
      break;
    }
    const item = {
      id: nextId++, file, name, size: file.size, srcUrl: URL.createObjectURL(file), status: 'analyzing',
      info: null, recommendation: null, result: null, progress: null, error: null, width: 0, height: 0,
    };
    state.items.push(item);
    if (added++ === 0) select(item.id);
    analyzeItem(item);
  }
  if (added) showWorkspace(true);
  notify();
}

async function analyzeItem(item) {
  try {
    const info = await analyzer.call('analyze', { file: item.file });
    if (!findItem(item.id)) return;
    item.info = info;
    item.width = info.analysis.width;
    item.height = info.analysis.height;
    item.recommendation = recommend(info.analysis, { exif: info.exif }, await capsReady);
    item.status = 'ready';
    warmUp(item.recommendation.settings);
  } catch (err) {
    if (!findItem(item.id)) return;
    item.status = 'error';
    item.error = err.message;
    toast(`${item.name}: ${err.message}`, 'bad', 6000);
  }
  notify();
}

const fileInput = $('#fileInput');
fileInput.addEventListener('change', () => {
  addFiles([...fileInput.files]);
  fileInput.value = '';
});
$('#pickBtn').addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});
const dropzone = $('#dropzone');
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.target === dropzone && (e.key === 'Enter' || e.key === ' ')) {
    e.preventDefault();
    fileInput.click();
  }
});

for (const sample of SAMPLES) {
  $('#sampleRow').append(h('button', {
    type: 'button',
    class: 'chip',
    onclick: async (e) => {
      e.stopPropagation();
      try {
        addFiles([await sample.make()]);
      } catch (err) {
        toast(err.message, 'bad');
      }
    },
  }, sample.label));
}

document.addEventListener('paste', (e) => {
  const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith('image/'));
  if (files.length) {
    e.preventDefault();
    addFiles(files);
  }
});

const overlay = $('#dropOverlay');
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
window.addEventListener('dragenter', (e) => {
  if (!hasFiles(e)) return;
  dragDepth++;
  overlay.hidden = false;
});
window.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) overlay.hidden = true;
});
window.addEventListener('dragover', (e) => {
  if (hasFiles(e)) e.preventDefault();
});
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  overlay.hidden = true;
  addFiles([...e.dataTransfer.files]);
});

// ---------- selection & lifecycle ----------
function showWorkspace(show) {
  $('#hero').hidden = show;
  $('#workspace').hidden = !show;
  document.body.classList.toggle('has-items', show);
}

function select(id) {
  state.selectedId = id;
  viewerKey = null;
  notify();
}

function removeItem(id) {
  const idx = state.items.findIndex((it) => it.id === id);
  if (idx < 0 || state.items[idx].status === 'processing') return;
  const [item] = state.items.splice(idx, 1);
  URL.revokeObjectURL(item.srcUrl);
  if (item.result) URL.revokeObjectURL(item.result.url);
  if (state.selectedId === id) {
    const next = state.items[Math.min(idx, state.items.length - 1)];
    if (next) select(next.id);
    else state.selectedId = null;
  }
  if (!state.items.length) {
    viewer.clear();
    showWorkspace(false);
  }
  notify();
}

function clearAll() {
  if (state.batchRunning) return;
  const unsaved = state.items.some((it) => it.result);
  if (unsaved && !window.confirm('Remove all images? Enhanced results you have not downloaded will be lost.')) return;
  for (const item of [...state.items]) if (item.status !== 'processing') removeItem(item.id);
}

// ---------- enhancement ----------
async function enhanceItem(item, settings) {
  item.status = 'processing';
  item.error = null;
  item.progress = { fraction: 0, label: 'Starting' };
  notify();
  try {
    const res = await enhancer.call('enhance', { file: item.file, settings, fetchNow: fetchNow() }, (fraction, label) => {
      item.progress = { fraction, label };
      notify();
    });
    if (!findItem(item.id)) return;
    if (item.result) URL.revokeObjectURL(item.result.url);
    item.result = { ...res, url: URL.createObjectURL(res.blob), settings };
    item.status = 'done';
    if (res.capped) toast(`${item.name}: output limited to 33 megapixels.`, 'info');
    for (const note of res.notes) toast(`${item.name}: ${note}`, 'warn', 7000);
    for (const model of res.pending) prefetch(model);
  } catch (err) {
    if (!findItem(item.id)) return;
    if (err.name === 'AbortError') {
      item.status = item.result ? 'done' : 'ready';
    } else {
      item.status = 'error';
      item.error = err.message;
      toast(`${item.name}: ${err.message}`, 'bad', 6000);
    }
  } finally {
    item.progress = null;
    notify();
    upgradePending();
  }
}

async function enhanceAll() {
  if (state.batchRunning) return;
  await checkLocalAI();
  const jobs = state.items
    .filter((it) => it.info && it.status !== 'processing')
    .map((it) => [it, settingsFor(it)])
    .filter(([it, s]) => !(it.status === 'done' && sameSettings(it.result?.settings, s)));
  if (!jobs.length) {
    toast('Every image is already enhanced with these settings.', 'info');
    return;
  }
  state.batchRunning = true;
  batchCancelled = false;
  for (const [it] of jobs) it.status = 'queued';
  notify();
  let done = 0;
  for (const [item, settings] of jobs) {
    if (batchCancelled) break;
    if (!findItem(item.id)) continue;
    await enhanceItem(item, settings);
    if (item.status === 'done') done++;
  }
  for (const [it] of jobs) if (it.status === 'queued') it.status = it.result ? 'done' : 'ready';
  state.batchRunning = false;
  notify();
  if (!batchCancelled) toast(`Enhanced ${done} of ${jobs.length} image${jobs.length === 1 ? '' : 's'}.`, 'good');
  upgradePending();
}

function cancelWork() {
  batchCancelled = true;
  enhancer.cancel();
  // The worker restarts, so loaded AI models have to load again next time.
  warmed.clear();
  // Cancelled work is not restarted by itself when a download finishes.
  for (const it of state.items) if (it.result) it.upgraded = new Set(it.result.pending);
  toast('Cancelled.', 'info', 2000);
}

// Gets the AI model the analysis picked ready, so the first run starts
// straight away: bundled models are loaded, large ones downloaded in the
// background. Failures are ignored here: the run falls back to a model that works.
const warmed = new Set();
function warmUp(settings) {
  // With the local AI server running, the browser loads no models at all.
  if (settings.engine !== 'ai' || state.localAI) return;
  if (AI_MODELS[settings.aiModel].url) {
    prefetch(settings.aiModel);
  } else if (!warmed.has(settings.aiModel)) {
    warmed.add(settings.aiModel);
    enhancer.call('aiPrepare', { model: settings.aiModel }).catch(() => warmed.delete(settings.aiModel));
  }
}

// Downloads a large model once (it is kept in the browser's cache). Images
// that were enhanced without it are re-run when it arrives.
function prefetch(model) {
  const current = state.downloads[model];
  if (current && current.status !== 'failed') return;
  const { name } = AI_MODELS[model] || FACE_MODELS[model];
  // 'checking' until bytes arrive: a model already in the cache loads quietly.
  state.downloads[model] = { status: 'checking', progress: 0 };
  notify();
  loader.call('fetchModel', { model }, (fraction) => {
    state.downloads[model] = { status: 'downloading', progress: fraction };
    notify();
  }).then(({ cached }) => {
    state.downloads[model] = { status: 'ready', progress: 1, cached };
    upgradePending();
  }).catch((err) => {
    state.downloads[model] = { status: 'failed', progress: 0, message: err.message };
    toast(`${name} could not be downloaded (${err.message}). Your results are kept without it.`, 'warn', 7000);
  }).finally(notify);
}

// Models that finished downloading but that this browser could not keep; a
// run may fetch these itself.
const fetchNow = () => Object.keys(state.downloads).filter((m) => state.downloads[m].status === 'ready' && !state.downloads[m].cached);

// Re-runs one finished image whose result had to skip a model that is now
// ready, once per model, whenever nothing else is running.
function upgradePending() {
  if (state.batchRunning || state.items.some((it) => it.status === 'processing' || it.status === 'queued')) return;
  for (const item of state.items) {
    if (item.status !== 'done') continue;
    const ready = (item.result?.pending ?? []).filter((m) => state.downloads[m]?.status === 'ready' && !item.upgraded?.has(m));
    if (!ready.length) continue;
    item.upgraded ??= new Set();
    for (const m of ready) item.upgraded.add(m);
    const names = ready.map((m) => (AI_MODELS[m] || FACE_MODELS[m]).name).join(' and ');
    toast(`${names} ready. Updating ${item.name}…`, 'info', 4000);
    enhanceItem(item, settingsFor(item));
    return;
  }
}

// ---------- export ----------
async function exportBlob(item) {
  const { format, quality } = state.output;
  if (format === 'png') return item.result.blob;
  const mime = format === 'jpeg' ? 'image/jpeg' : 'image/webp';
  const { blob } = await analyzer.call('convert', { blob: item.result.blob, mime, quality: quality / 100 });
  if (blob.type !== mime) toast(`This browser cannot save ${format.toUpperCase()}; saved as PNG instead.`, 'warn');
  return blob;
}

const outputName = (item, blob) =>
  `${item.name.replace(/\.[^.]+$/, '') || 'image'}-pixellift-${item.result.width}x${item.result.height}.${EXT_BY_TYPE[blob.type] || 'png'}`;

async function downloadSelected() {
  const item = selectedItem();
  if (!item?.result) return;
  try {
    const blob = await exportBlob(item);
    download(blob, outputName(item, blob));
  } catch (err) {
    toast(err.message, 'bad');
  }
}

async function downloadAll() {
  const items = state.items.filter((it) => it.status === 'done' && it.result);
  if (!items.length) return;
  toast(`Preparing ${items.length} image${items.length === 1 ? '' : 's'}…`, 'info', 2500);
  try {
    const files = [];
    for (const item of items) {
      const blob = await exportBlob(item);
      files.push({ name: outputName(item, blob), data: new Uint8Array(await blob.arrayBuffer()) });
    }
    download(new Blob(zipParts(files), { type: 'application/zip' }), `pixellift-${items.length}-images.zip`);
  } catch (err) {
    toast(err.message, 'bad');
  }
}

// ---------- render ----------
const pill = $('#stagePill');
const stageInfo = $('#stageInfo');

subscribe((st) => {
  const item = selectedItem();
  const locked = st.items.some((it) => it.status === 'processing');
  const settings = settingsFor(item);
  const stale = Boolean(item?.result) && !sameSettings(item.result.settings, settings);
  controls.render(st, item, { settings, stale, locked });
  analysisView.render(item);
  queue.render(st);

  if (!item?.width) {
    if (viewerKey !== '') viewer.clear();
    viewerKey = '';
  } else {
    const key = `${item.id}|${item.result?.url || ''}`;
    if (key !== viewerKey) {
      const reset = !viewerKey || !viewerKey.startsWith(`${item.id}|`);
      viewerKey = key;
      const r = item.result;
      viewer.show({ beforeUrl: item.srcUrl, afterUrl: r?.url, width: r ? r.width : item.width, height: r ? r.height : item.height, reset });
    }
  }

  const busy = item?.status === 'analyzing' || item?.status === 'processing';
  pill.hidden = !busy;
  if (busy) {
    const label = item.status === 'analyzing' ? 'Analyzing…' : `${item.progress?.label || 'Working'} · ${Math.round((item.progress?.fraction || 0) * 100)}%`;
    pill.replaceChildren(h('span', { class: 'spinner' }), label);
  }
  stageInfo.textContent = !item?.width
    ? item?.name || ''
    : item.result
      ? `Original ${item.width}×${item.height} · Enhanced ${item.result.width}×${item.result.height}`
      : `Original ${item.width}×${item.height}`;
});

notify();

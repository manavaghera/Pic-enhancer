// The "Enhance" panel. The user only picks the output quality; the AI's plan
// for the selected image is shown, then the run / download footer. Built once,
// then updated in place.

import { FACE_MODELS } from '../ai/faceModels.js';
import { AI_MODELS } from '../ai/models.js';
import { aiInputSize, computeTargetSize } from '../core/pipeline.js';
import { describePlan } from '../core/recommend.js';
import { formatBytes, formatDims, h } from './dom.js';
import { icons } from './icons.js';

const QUALITY_OPTIONS = [['auto', 'Auto'], ['1x', '1×'], ['2x', '2×'], ['4x', '4×'], ['4k', '4K'], ['8k', '8K']];

function segmented(label, options, onPick) {
  const buttons = options.map(([value, text]) =>
    h('button', { type: 'button', class: 'seg-btn', 'data-value': value, 'aria-pressed': 'false', onclick: () => onPick(value) }, text),
  );
  return { el: h('div', { class: 'segmented', role: 'group', 'aria-label': label }, buttons), buttons };
}

const planItem = ({ title, detail, level = 'good' }) =>
  h('li', { class: `issue issue-${level}` }, h('span', { class: 'issue-dot' }), h('div', {}, h('strong', {}, title), h('p', {}, detail)));

// Adds the state of a step's downloaded model: still downloading (the result
// so far was made without it and updates by itself), failed, or skipped.
function withDownload(step, result, downloads) {
  const waiting = Boolean(step.model && result?.pending?.includes(step.model));
  if (step.title === 'Face restoration' && result?.faces === 0 && !waiting) {
    return { ...step, detail: 'No human faces needed fixing in this image (animal faces are left as they are).' };
  }
  if (!step.model) return step;
  const d = downloads[step.model];
  const { size } = AI_MODELS[step.model] || FACE_MODELS[step.model];
  const isFace = Boolean(FACE_MODELS[step.model]);
  if (d?.status === 'downloading') {
    const until = isFace ? 'Faces are fixed' : 'The fast model is used until then and the image is updated';
    return { ...step, level: 'info', detail: `Downloading this model once: ${Math.round(d.progress * 100)}% of ${size}. ${until} automatically when it is done.` };
  }
  if (d?.status === 'failed' && waiting) return { ...step, level: 'warn', detail: `The model could not be downloaded, so this step was skipped. ${d.message}` };
  return step;
}

export function createControls(root, actions) {
  const quality = segmented('Output quality', QUALITY_OPTIONS, (v) => actions.onQuality(v));
  const sizeNote = h('p', { class: 'hint size-note' });
  const planList = h('ul', { class: 'issues' });
  const planNote = h('p', { class: 'hint' });

  const progressBar = h('div', { class: 'progress-bar' });
  const progressLabel = h('span', { class: 'progress-label' });
  const progressPct = h('span', { class: 'progress-pct' });
  const progress = h('div', { class: 'progress', hidden: true },
    h('div', { class: 'progress-track' }, progressBar),
    h('div', { class: 'progress-meta' }, progressLabel, progressPct));
  const enhanceBtn = h('button', { type: 'button', class: 'btn btn-primary btn-lg', onclick: () => actions.onEnhance() });
  const cancelBtn = h('button', { type: 'button', class: 'btn btn-ghost', hidden: true, onclick: () => actions.onCancel() }, 'Cancel');
  const staleNote = h('p', { class: 'hint stale-note', hidden: true }, 'Quality changed since the last run.');

  const resultInfo = h('p', { class: 'result-info' });
  const format = h('select', { class: 'select', 'aria-label': 'Download format' },
    h('option', { value: 'png' }, 'PNG · lossless'),
    h('option', { value: 'jpeg' }, 'JPG'),
    h('option', { value: 'webp' }, 'WebP'));
  format.addEventListener('change', () => actions.onOutput('format', format.value));
  const fileQuality = h('input', { type: 'range', min: 50, max: 100, step: 1, 'aria-label': 'Download quality' });
  const fileQualityOut = h('output', { class: 'slider-val' });
  fileQuality.addEventListener('input', () => actions.onOutput('quality', Number(fileQuality.value)));
  const fileQualityRow = h('div', { class: 'slider compact' },
    h('div', { class: 'slider-head' }, h('label', {}, 'File quality'), fileQualityOut), fileQuality);
  const downloadBtn = h('button', { type: 'button', class: 'btn btn-accent', onclick: () => actions.onDownload() },
    h('span', { html: icons.download }), 'Download');
  const result = h('div', { class: 'result-box', hidden: true },
    resultInfo,
    h('div', { class: 'download-row' }, format, downloadBtn),
    fileQualityRow);

  root.append(
    h('div', { class: 'panel-scroll' },
      h('section', { class: 'psec' }, h('h3', { class: 'psec-title' }, 'Quality'), quality.el, sizeNote),
      h('section', { class: 'psec' }, h('h3', { class: 'psec-title' }, 'What AI will do'), planNote, planList),
    ),
    h('div', { class: 'panel-footer' }, progress, staleNote, h('div', { class: 'run-row' }, enhanceBtn, cancelBtn), result),
  );

  let planKey = null;

  // ctx.settings: the effective settings for the selected image.
  // ctx.stale: the item's result was made with different settings.
  // ctx.locked: another image is being processed right now.
  function render(state, item, ctx = {}) {
    const s = ctx.settings;
    for (const b of quality.buttons) b.setAttribute('aria-pressed', String(b.dataset.value === state.quality));

    if (item?.width) {
      const t = computeTargetSize(item.width, item.height, s.scale, { integer: s.resample === 'nearest' });
      let note = `${formatDims(item.width, item.height)} → ${formatDims(t.width, t.height)}`;
      if (state.quality === 'auto') note += ' · size picked for this image';
      if (t.capped) note += ' · capped at 33 MP';
      else if (t.factor === 1 && s.scale !== '1x') note += ' · already this size';
      sizeNote.textContent = note;
    } else {
      sizeNote.textContent = '';
    }

    const ready = Boolean(item?.recommendation);
    const steps = ready ? describePlan(s).map((st) => withDownload(st, item.result, state.downloads)) : [];
    if (ready && s.faces > 0 && !s.refine) {
      steps.splice(1, 0, { title: 'Realistic detail (off)', detail: 'Start the local AI server on this PC for Artguru-like detail.', level: 'info' });
    }
    const key = JSON.stringify(steps);
    if (key !== planKey) {
      planKey = key;
      planList.replaceChildren(...steps.map(planItem));
    }
    if (!item) planNote.textContent = 'Add an image and the AI picks the upscaler, face fixes, cleanup and colour for it.';
    else if (item.status === 'analyzing') planNote.textContent = 'Analyzing the image…';
    else if (!ready) planNote.textContent = item.error || '';
    else {
      const [aw, ah] = aiInputSize(item.width, item.height, s);
      const smaller = s.engine === 'ai' && s.resample !== 'nearest' && aw < item.width;
      planNote.textContent = `Picked from this image's analysis${smaller ? `; AI works on a ${formatDims(aw, ah)} copy` : ''}.`;
    }

    const busy = item?.status === 'processing';
    progress.hidden = !busy;
    cancelBtn.hidden = !busy && !state.batchRunning;
    enhanceBtn.disabled = !ready || busy || state.batchRunning || ctx.locked;
    enhanceBtn.replaceChildren(h('span', { html: icons.sparkle }), item?.result ? 'Enhance again' : 'Enhance image');
    if (busy) {
      const f = item.progress?.fraction || 0;
      progressBar.style.width = `${Math.round(f * 100)}%`;
      progressLabel.textContent = item.progress?.label || 'Working…';
      progressPct.textContent = `${Math.round(f * 100)}%`;
    }

    const res = item?.result;
    result.hidden = !res || busy;
    staleNote.hidden = !res || busy || !ctx.stale;
    if (res) {
      const faces = res.faces > 0 ? ` · ${res.faces} face${res.faces === 1 ? '' : 's'} restored` : '';
      resultInfo.textContent = `${formatDims(res.width, res.height)} · ${formatBytes(res.blob.size)} PNG · ${(res.ms / 1000).toFixed(1)} s${faces}`;
      format.value = state.output.format;
      fileQuality.value = String(state.output.quality);
      fileQualityOut.textContent = `${state.output.quality}`;
      fileQuality.style.setProperty('--fill', `${((state.output.quality - 50) / 50) * 100}%`);
      fileQualityRow.hidden = state.output.format === 'png';
    }
  }

  return { render };
}

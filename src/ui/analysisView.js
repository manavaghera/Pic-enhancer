// The "Analyze" panel: quality score, sub-scores (before vs. after), findings,
// histogram, palette, technical details and camera metadata.

import { formatBytes, formatDims, h } from './dom.js';
import { toast } from './toast.js';

const RING = 2 * Math.PI * 34;

function ring(value, caption, tone = 'accent') {
  return h('div', { class: 'ring-wrap' },
    h('div', {
      class: `ring ring-${tone}`,
      html: `<svg viewBox="0 0 80 80" aria-hidden="true"><circle cx="40" cy="40" r="34" class="ring-bg"/><circle cx="40" cy="40" r="34" class="ring-fg" stroke-dasharray="${(value / 100) * RING} ${RING}"/></svg>`,
    }, h('span', { class: 'ring-value' }, String(value))),
    h('span', { class: 'ring-caption' }, caption));
}

const tone = (v) => (v >= 75 ? 'good' : v >= 50 ? 'warn' : 'bad');

function subScores(a, b) {
  const rows = [
    ['Sharpness', (x) => x.sharpness.score],
    ['Low noise', (x) => x.noise.score],
    ['Exposure', (x) => x.exposure.score],
    ['Contrast', (x) => x.contrastScore],
    ['Resolution', (x) => x.resolution.score],
    ['Compression', (x) => x.blockiness.score],
  ];
  return h('div', { class: 'bars' }, rows.map(([label, get]) => {
    const v = get(a);
    const after = b ? get(b) : null;
    return h('div', { class: 'bar-row' },
      h('span', { class: 'bar-label' }, label),
      h('div', { class: 'bar-track' },
        h('div', { class: `bar-fill bar-${tone(v)}`, style: { width: `${v}%` } }),
        after != null && h('div', { class: 'bar-after', style: { left: `${after}%` }, title: `After: ${after}` })),
      h('span', { class: 'bar-val' }, after != null ? `${v} → ${after}` : String(v)));
  }));
}

function drawHistogram(canvas, hist, mode) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 300;
  const H = canvas.clientHeight || 110;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const chans = mode === 'luma'
    ? [['l', 'rgba(140,146,170,0.85)']]
    : [['r', 'rgba(239,68,68,0.5)'], ['g', 'rgba(34,197,94,0.5)'], ['b', 'rgba(59,130,246,0.5)']];
  let max = 1;
  for (const [c] of chans) for (let i = 1; i < 255; i++) max = Math.max(max, hist[c][i]);
  for (const [c, color] of chans) {
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let i = 0; i < 256; i++) ctx.lineTo((i / 255) * W, H - Math.min(1, hist[c][i] / max) * (H - 4));
    ctx.lineTo(W, H);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }
}

const colorfulnessLabel = (c) =>
  c < 15 ? 'Not colorful' : c < 33 ? 'Slightly colorful' : c < 45 ? 'Moderately colorful' : c < 59 ? 'Colorful' : c < 82 ? 'Quite colorful' : 'Highly colorful';

const table = (rows) =>
  h('dl', { class: 'kv' }, rows.filter(Boolean).map(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]));

function details(item) {
  const { analysis: a, format, jpegQuality } = item.info;
  return table([
    ['Dimensions', formatDims(a.width, a.height)],
    ['Megapixels', `${a.megapixels.toFixed(2)} MP · ${a.resolution.tier}`],
    ['Aspect ratio', a.aspect],
    ['Format', jpegQuality ? `${format} · quality ≈ ${jpegQuality}` : format],
    ['File size', formatBytes(item.size)],
    ['Print size', `${(a.width / 300).toFixed(1)} × ${(a.height / 300).toFixed(1)} in at 300 DPI`],
    ['Brightness', `${Math.round(a.brightness)} / 255 · ${a.exposure.label}`],
    ['Contrast (RMS)', Math.round(a.contrast)],
    ['Dynamic range', `${a.dynamicRange} / 255`],
    ['Saturation', `${Math.round(a.saturation * 100)}%`],
    ['Colorfulness', `${Math.round(a.colorfulness)} · ${a.isGrayscale ? 'Grayscale' : colorfulnessLabel(a.colorfulness)}`],
    ['Color cast', a.cast.label === 'Neutral' ? 'None detected' : `${a.cast.label} (${Math.round(a.cast.strength * 100)}%)`],
    ['Unique colors', a.uniqueColors > 4096 ? '4,096+' : a.uniqueColors.toLocaleString()],
    ['Transparency', a.hasAlpha ? `${(a.transparentPct * 100).toFixed(1)}% transparent` : 'None'],
    ['Noise level', `σ ≈ ${a.noise.sigma.toFixed(2)}`],
    ['Edge width', a.sharpness.edgeWidth == null ? 'No distinct edges' : `${a.sharpness.edgeWidth.toFixed(1)} px`],
    ['Edge energy', Math.round(a.sharpness.variance).toLocaleString()],
    ['JPEG blockiness', `${a.blockiness.ratio.toFixed(2)}×`],
  ]);
}

function camera(exif) {
  if (!exif) return null;
  const exposure = exif.exposureTime
    ? exif.exposureTime < 1 ? `1/${Math.round(1 / exif.exposureTime)} s` : `${exif.exposureTime} s`
    : null;
  const focal = exif.focalLength
    ? `${Math.round(exif.focalLength)} mm${exif.focalLength35mm ? ` (${exif.focalLength35mm} mm equiv.)` : ''}`
    : null;
  const rows = [
    (exif.make || exif.model) && ['Camera', [exif.make, exif.model].filter(Boolean).join(' ')],
    exif.lensModel && ['Lens', exif.lensModel],
    (exif.dateTimeOriginal || exif.dateTime) && ['Taken', exif.dateTimeOriginal || exif.dateTime],
    exposure && ['Shutter', exposure],
    exif.fNumber && ['Aperture', `f/${exif.fNumber.toFixed(1)}`],
    exif.iso && ['ISO', exif.iso],
    focal && ['Focal length', focal],
    exif.flash != null && ['Flash', exif.flash & 1 ? 'Fired' : 'Did not fire'],
    exif.software && ['Software', exif.software],
    exif.hasGps && ['GPS', 'Location embedded'],
  ];
  return rows.some(Boolean) ? table(rows) : null;
}

function palette(colors) {
  return h('div', { class: 'palette' }, colors.map((c) =>
    h('button', {
      type: 'button',
      class: 'swatch',
      title: `Copy ${c.hex}`,
      onclick: async () => {
        try {
          await navigator.clipboard.writeText(c.hex);
          toast(`Copied ${c.hex}`, 'good', 1800);
        } catch {
          toast(c.hex, 'info', 2500);
        }
      },
    },
    h('span', { class: 'swatch-color', style: { background: c.hex } }),
    h('span', { class: 'swatch-hex' }, c.hex),
    h('span', { class: 'swatch-pct' }, `${Math.round(c.pct * 100)}%`))));
}

export function createAnalysisView(root) {
  let key = null;
  let histMode = 'rgb';
  let histSource = 'before';
  let lastRedraw = null;

  function build(item) {
    if (!item) return h('p', { class: 'empty-note' }, 'Add an image to see its analysis.');
    if (item.status === 'analyzing' || !item.info) {
      if (item.status === 'error') return h('p', { class: 'error-note' }, item.error);
      return h('div', { class: 'analyzing' }, h('span', { class: 'spinner' }), 'Analyzing image…');
    }
    const a = item.info.analysis;
    const b = item.result?.analysis || null;
    if (!b) histSource = 'before';

    const canvas = h('canvas', { class: 'histogram', 'aria-label': 'Histogram' });
    const redraw = () => drawHistogram(canvas, (histSource === 'after' && b ? b : a).histogram, histMode);
    const toggle = (options, get, set) => h('div', { class: 'mini-seg' }, options.map(([v, label]) =>
      h('button', {
        type: 'button',
        'aria-pressed': String(get() === v),
        onclick: (e) => {
          set(v);
          for (const btn of e.currentTarget.parentElement.children) btn.setAttribute('aria-pressed', String(btn === e.currentTarget));
          redraw();
        },
      }, label)));
    lastRedraw = redraw;
    requestAnimationFrame(redraw);

    const issues = item.recommendation?.issues || [];
    const cam = camera(item.info.exif);
    return h('div', { class: 'analysis' },
      h('div', { class: 'score-card' },
        ring(a.score, b ? 'Before' : a.grade, tone(a.score)),
        b && ring(b.score, 'After', tone(b.score)),
        h('div', { class: 'score-text' },
          h('strong', {}, b ? `${b.grade} after enhancement` : `${a.grade} quality`),
          h('span', {}, b
            ? `Score ${b.score - a.score >= 0 ? '+' : ''}${b.score - a.score} · ${formatDims(b.width, b.height)}`
            : `${formatDims(a.width, a.height)} · ${a.megapixels.toFixed(1)} MP`))),
      subScores(a, b),
      h('section', { class: 'psec' },
        h('h3', { class: 'psec-title' }, 'Findings'),
        h('ul', { class: 'issues' }, issues.map((it) =>
          h('li', { class: `issue issue-${it.level}` },
            h('span', { class: 'issue-dot' }),
            h('div', {}, h('strong', {}, it.title), h('p', {}, it.detail)))))),
      h('section', { class: 'psec' },
        h('div', { class: 'psec-head' },
          h('h3', { class: 'psec-title' }, 'Histogram'),
          h('div', { class: 'psec-tools' },
            b && toggle([['before', 'Before'], ['after', 'After']], () => histSource, (v) => { histSource = v; }),
            toggle([['rgb', 'RGB'], ['luma', 'Luma']], () => histMode, (v) => { histMode = v; }))),
        canvas),
      a.palette.length > 0 && h('section', { class: 'psec' }, h('h3', { class: 'psec-title' }, 'Dominant colors'), palette(a.palette)),
      h('section', { class: 'psec' }, h('h3', { class: 'psec-title' }, 'Details'), details(item)),
      cam && h('section', { class: 'psec' }, h('h3', { class: 'psec-title' }, 'Camera'), cam));
  }

  return {
    render(item) {
      const k = item ? `${item.id}|${item.status}|${item.info ? 1 : 0}|${item.result?.url || ''}` : '';
      if (k === key) return;
      key = k;
      lastRedraw = null;
      root.replaceChildren(build(item));
    },
    // Redraws the histogram at its real size once the tab becomes visible.
    refresh() {
      lastRedraw?.();
    },
  };
}

// Batch strip: one card per image (keyed, updated in place) plus batch actions.

import { formatDims, h } from './dom.js';
import { icons } from './icons.js';

function statusText(item) {
  switch (item.status) {
    case 'analyzing': return 'Analyzing…';
    case 'queued': return 'Queued';
    case 'processing': return `${Math.round((item.progress?.fraction || 0) * 100)}%`;
    case 'done': return `✓ ${formatDims(item.result.width, item.result.height)}`;
    case 'error': return 'Failed';
    default: return item.info ? `Score ${item.info.analysis.score}` : 'Ready';
  }
}

export function createQueue(root, actions) {
  const list = h('div', { class: 'queue-list', role: 'listbox', 'aria-label': 'Images' });
  const addBtn = h('button', { type: 'button', class: 'queue-add', title: 'Add images', 'aria-label': 'Add images', html: icons.plus, onclick: actions.onAdd });
  const count = h('span', { class: 'queue-count' });
  const enhanceAll = h('button', { type: 'button', class: 'btn btn-soft btn-sm', onclick: actions.onEnhanceAll });
  const downloadAll = h('button', { type: 'button', class: 'btn btn-soft btn-sm', onclick: actions.onDownloadAll },
    h('span', { html: icons.zip }), 'Download all');
  const clear = h('button', { type: 'button', class: 'btn btn-ghost btn-sm', onclick: actions.onClear, title: 'Remove all images' },
    h('span', { html: icons.trash }), 'Clear');
  root.append(
    h('div', { class: 'queue-head' }, count, h('div', { class: 'queue-actions' }, enhanceAll, downloadAll, clear)),
    h('div', { class: 'queue-body' }, list, addBtn),
  );
  const cards = new Map();

  function card(item) {
    const status = h('span', { class: 'qc-status' });
    const bar = h('div', { class: 'qc-bar' });
    const remove = h('button', {
      type: 'button',
      class: 'qc-remove',
      'aria-label': `Remove ${item.name}`,
      html: icons.x,
      onclick: (e) => {
        e.stopPropagation();
        actions.onRemove(item.id);
      },
    });
    const el = h('div', {
      class: 'qcard',
      role: 'option',
      tabindex: 0,
      title: item.name,
      onclick: () => actions.onSelect(item.id),
      onkeydown: (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          actions.onSelect(item.id);
        }
      },
    },
    h('div', { class: 'qc-thumb' }, h('img', { src: item.srcUrl, alt: '', draggable: 'false' }), h('div', { class: 'qc-progress' }, bar)),
    h('div', { class: 'qc-meta' }, h('span', { class: 'qc-name' }, item.name), status),
    remove);
    return { el, status, bar, remove };
  }

  function render(state) {
    const ids = new Set(state.items.map((it) => it.id));
    for (const [id, c] of cards) {
      if (!ids.has(id)) {
        c.el.remove();
        cards.delete(id);
      }
    }
    state.items.forEach((item, i) => {
      let c = cards.get(item.id);
      if (!c) {
        c = card(item);
        cards.set(item.id, c);
      }
      if (list.children[i] !== c.el) list.insertBefore(c.el, list.children[i] || null);
      c.el.setAttribute('aria-selected', String(item.id === state.selectedId));
      c.el.dataset.status = item.status;
      c.status.textContent = statusText(item);
      c.bar.style.width = item.status === 'processing' ? `${Math.round((item.progress?.fraction || 0) * 100)}%` : '0%';
      c.remove.disabled = item.status === 'processing';
    });

    const total = state.items.length;
    const done = state.items.filter((it) => it.status === 'done').length;
    const runnable = state.items.filter((it) => it.info && it.status !== 'processing').length;
    count.textContent = `${total} image${total === 1 ? '' : 's'}${done ? ` · ${done} enhanced` : ''}`;
    enhanceAll.disabled = state.batchRunning || runnable === 0;
    enhanceAll.replaceChildren(h('span', { html: icons.sparkle }), state.batchRunning ? 'Enhancing…' : 'Enhance all');
    downloadAll.disabled = done === 0 || state.batchRunning;
    clear.disabled = state.batchRunning;
  }

  return { render };
}

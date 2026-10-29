import { h } from './dom.js';

let region;

export function toast(message, kind = 'info', ms = 4200) {
  if (!region) {
    region = h('div', { class: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.append(region);
  }
  const el = h('div', { class: `toast toast-${kind}` }, message);
  region.append(el);
  setTimeout(() => {
    el.classList.add('leaving');
    setTimeout(() => el.remove(), 250);
  }, ms);
}

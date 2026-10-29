// ONNX Runtime Web, shared by face restoration and the HQ upscaler. Large
// models are fetched from Hugging Face on first use, checked against a pinned
// SHA-256 and kept in Cache Storage. Loaded on demand inside the worker.

import * as ort from 'onnxruntime-web/webgpu';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';

export { ort };

ort.env.wasm.wasmPaths = { wasm: new URL(wasmUrl, self.location.href).href };
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CACHE = 'pixellift-models-v1';
// A download that receives nothing for this long is treated as failed.
const STALL_MS = 30_000;

const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');

// Browsers reject very large single Cache Storage entries (Chromium fails a
// 340 MB put), so the model is stored in parts plus a marker written last.
const PART = 32 * 1024 * 1024;
const partKey = (url, i) => `${url}?pixellift-part=${i}`;
const doneKey = (url) => `${url}?pixellift-complete`;

async function cacheRead(cache, url, bytes) {
  if (!(await cache.match(doneKey(url)))) return null;
  const out = new Uint8Array(bytes);
  for (let i = 0, o = 0; o < bytes; i++, o += PART) {
    const hit = await cache.match(partKey(url, i));
    if (!hit) return null;
    const part = new Uint8Array(await hit.arrayBuffer());
    if (part.length !== Math.min(PART, bytes - o)) return null;
    out.set(part, o);
  }
  return out;
}

async function cacheWrite(cache, url, data) {
  for (let i = 0, o = 0; o < data.length; i++, o += PART) {
    await cache.put(partKey(url, i), new Response(data.slice(o, o + PART)));
  }
  await cache.put(doneKey(url), new Response('ok'));
}

export async function isCached({ url }) {
  try {
    return Boolean(await (await caches.open(CACHE)).match(doneKey(url)));
  } catch {
    return false;
  }
}

// cachedOnly: fail with code NOT_CACHED instead of downloading, so a run never
// waits on a large download (the app fetches models in the background).
export async function downloadVerified({ name, url, sha256, bytes }, onProgress, { cachedOnly = false } = {}) {
  let cache = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cacheRead(cache, url, bytes);
    if (hit) return hit;
  } catch {
    cache = null;
  }
  if (cachedOnly) throw Object.assign(new Error(`${name} is not downloaded yet.`), { code: 'NOT_CACHED' });
  const abort = new AbortController();
  let timer = setTimeout(() => abort.abort(), STALL_MS);
  const out = new Uint8Array(bytes);
  let got = 0;
  try {
    const res = await fetch(url, { signal: abort.signal });
    if (!res.ok || !res.body) throw new Error(`${name} download failed (HTTP ${res.status}).`);
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      clearTimeout(timer);
      timer = setTimeout(() => abort.abort(), STALL_MS);
      if (got + value.length > bytes) throw new Error(`${name} download is larger than expected.`);
      out.set(value, got);
      got += value.length;
      onProgress?.(got / bytes);
    }
  } catch (err) {
    if (abort.signal.aborted) throw new Error(`${name} download stalled. Check your connection and try again.`);
    if (err instanceof TypeError) throw new Error(`Could not download ${name}. Check your connection and try again.`);
    throw err;
  } finally {
    clearTimeout(timer);
  }
  if (got !== bytes) throw new Error(`${name} download was incomplete. Please try again.`);
  if (hex(await crypto.subtle.digest('SHA-256', out)) !== sha256) {
    throw new Error(`${name} failed its integrity check and was discarded.`);
  }
  try {
    if (cache) await cacheWrite(cache, url, out);
  } catch {
    /* storage full: it simply downloads again next time */
  }
  return out;
}

// Tries each backend in order; returns { session, backend }.
export async function createSession(bytes, backends = ['webgpu', 'wasm']) {
  let error = new Error('This browser has no backend that can run the model.');
  for (const backend of backends) {
    if (backend === 'webgpu' && !self.navigator?.gpu) continue;
    try {
      return { session: await ort.InferenceSession.create(bytes, { executionProviders: [backend] }), backend };
    } catch (err) {
      error = err;
    }
  }
  throw error;
}

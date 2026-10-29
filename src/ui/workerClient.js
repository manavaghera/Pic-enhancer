// Promise wrapper around a worker. cancel() terminates the worker (the only
// way to stop a long synchronous job) and starts a fresh one.

export class WorkerClient {
  constructor(factory) {
    this.factory = factory;
    this.pending = new Map();
    this.seq = 0;
    this.spawn();
  }

  spawn() {
    this.worker = this.factory();
    this.worker.onmessage = (e) => this.handle(e.data);
    this.worker.onerror = (e) => {
      e.preventDefault?.();
      this.failAll(new Error('The image processor crashed, most likely out of memory. Try a smaller output size.'));
      this.worker.terminate();
      this.spawn();
    };
  }

  call(type, payload, onProgress) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, type, payload });
    });
  }

  handle(msg) {
    const job = this.pending.get(msg?.id);
    if (!job) return;
    if (msg.type === 'progress') {
      job.onProgress?.(msg.fraction, msg.label);
      return;
    }
    this.pending.delete(msg.id);
    if (msg.type === 'result') job.resolve(msg.result);
    else job.reject(new Error(msg.message));
  }

  failAll(err) {
    for (const job of this.pending.values()) job.reject(err);
    this.pending.clear();
  }

  cancel() {
    this.worker.terminate();
    const err = new Error('Cancelled');
    err.name = 'AbortError';
    this.failAll(err);
    this.spawn();
  }
}

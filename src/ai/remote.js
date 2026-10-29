// Stand-ins for ONNX Runtime's session and tensor that run a model on the
// local AI server (server/app.py) instead of in the browser. While that server
// runs, all AI work goes there: Chrome's WebGPU and the server's CUDA sharing
// one GPU crashed the graphics driver. Only what this app uses is provided,
// and ONNX Runtime itself is never loaded, so the browser stays off the GPU.

import { LOCAL_AI_URL } from './models.js';

export const remoteOrt = {
  Tensor: class {
    constructor(type, data, dims) {
      Object.assign(this, { type, data, dims });
    }
  },
};

// The app's models all take "input" (CodeFormer also a scalar "weight");
// outputs come back under their own names.
export function remoteSession(model) {
  return {
    inputNames: ['input'],
    outputNames: ['output'],
    async run(feeds) {
      const x = feeds.input;
      const query = new URLSearchParams({ shape: x.dims.join(',') });
      if (feeds.weight) query.set('weight', String(feeds.weight.data[0]));
      const res = await fetch(`${LOCAL_AI_URL}/run/${model}?${query}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: x.data,
      });
      if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail || `Local AI server error (HTTP ${res.status}).`);
      const outputs = JSON.parse(res.headers.get('X-Outputs'));
      const buf = await res.arrayBuffer();
      const result = {};
      let offset = 0;
      for (const { name, shape } of outputs) {
        const n = shape.reduce((a, b) => a * b, 1);
        result[name] = { data: new Float32Array(buf, offset * 4, n), dims: shape };
        offset += n;
      }
      return result;
    },
    async release() {},
  };
}

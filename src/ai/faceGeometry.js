// Face alignment for GFPGAN: fit a similarity transform from five detected
// landmarks to the FFHQ template, sample an aligned 512x512 crop, and blend
// the restored crop back through the inverse mapping with a soft oval mask.

import { edgeWidth } from '../core/analyze.js';

// FFHQ five-point template for 512x512 crops, in image order: left eye,
// right eye, nose tip, left mouth corner, right mouth corner.
export const FFHQ_512 = [
  [192.98138, 239.94708],
  [318.90277, 240.1936],
  [256.63416, 314.01935],
  [201.26117, 371.41043],
  [313.08905, 371.15118],
];

// Least-squares similarity transform mapping src points onto dst points.
// Returns [a, b, tx, ty]: u = a*x - b*y + tx, v = b*x + a*y + ty.
export function similarityTransform(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, du = 0, dv = 0;
  for (let i = 0; i < n; i++) {
    sx += src[i][0];
    sy += src[i][1];
    du += dst[i][0];
    dv += dst[i][1];
  }
  sx /= n; sy /= n; du /= n; dv /= n;
  let num1 = 0, num2 = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i][0] - sx, y = src[i][1] - sy;
    const u = dst[i][0] - du, v = dst[i][1] - dv;
    num1 += x * u + y * v;
    num2 += x * v - y * u;
    den += x * x + y * y;
  }
  if (den === 0) throw new Error('Degenerate landmarks');
  const a = num1 / den, b = num2 / den;
  return [a, b, du - a * sx + b * sy, dv - b * sx - a * sy];
}

export const applyTransform = ([a, b, tx, ty], x, y) => [a * x - b * y + tx, b * x + a * y + ty];

export function invertTransform([a, b, tx, ty]) {
  const d = a * a + b * b;
  return [a / d, -b / d, -(a * tx + b * ty) / d, (b * tx - a * ty) / d];
}

// Bilinear sample of RGBA data at (x, y) with clamp-to-edge; writes 3 values.
function sampleRGB(data, w, h, x, y, out) {
  const xc = Math.min(w - 1, Math.max(0, x)), yc = Math.min(h - 1, Math.max(0, y));
  const x0 = Math.floor(xc), y0 = Math.floor(yc);
  const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
  const fx = xc - x0, fy = yc - y0;
  const p00 = (y0 * w + x0) * 4, p10 = (y0 * w + x1) * 4, p01 = (y1 * w + x0) * 4, p11 = (y1 * w + x1) * 4;
  for (let c = 0; c < 3; c++) {
    const top = data[p00 + c] + (data[p10 + c] - data[p00 + c]) * fx;
    const bot = data[p01 + c] + (data[p11 + c] - data[p01 + c]) * fx;
    out[c] = top + (bot - top) * fy;
  }
}

// Aligned crop as a CHW RGB tensor normalised to [-1, 1] (GFPGAN's input).
export function cropToTensor(img, m, size) {
  const inv = invertTransform(m);
  const plane = size * size;
  const t = new Float32Array(3 * plane);
  const px = [0, 0, 0];
  for (let v = 0; v < size; v++) {
    for (let u = 0; u < size; u++) {
      const [x, y] = applyTransform(inv, u, v);
      sampleRGB(img.data, img.width, img.height, x, y, px);
      const i = v * size + u;
      t[i] = px[0] / 127.5 - 1;
      t[plane + i] = px[1] / 127.5 - 1;
      t[2 * plane + i] = px[2] / 127.5 - 1;
    }
  }
  return t;
}

// Soft oval over the head (face plus most of the hair; eyes ~y240, mouth
// ~y371 in template space). The wide fade hides the seam between the
// restored head and its surroundings and spares neighbouring faces.
function faceMask(u, v, size) {
  const k = size / 512;
  const e = Math.hypot((u - 256 * k) / (185 * k), (v - 280 * k) / (240 * k));
  const t = Math.min(1, Math.max(0, (1 - e) / 0.35));
  return t * t * (3 - 2 * t);
}

// Bilinear sample of channel c of a CHW tensor in [-1, 1], returned as 0..255.
function sampleTensor(tensor, size, u, v, c) {
  const u0 = Math.floor(u), v0 = Math.floor(v);
  const u1 = Math.min(size - 1, u0 + 1), v1 = Math.min(size - 1, v0 + 1);
  const fu = u - u0, fv = v - v0;
  const o = c * size * size;
  const top = tensor[o + v0 * size + u0] + (tensor[o + v0 * size + u1] - tensor[o + v0 * size + u0]) * fu;
  const bot = tensor[o + v1 * size + u0] + (tensor[o + v1 * size + u1] - tensor[o + v1 * size + u0]) * fu;
  return (top + (bot - top) * fv + 1) * 127.5;
}

// Blends a restored CHW tensor in [-1, 1] back into img (in place). The face
// is first shifted to the average colour of the image under the mask core,
// because the background may have been colour-corrected and upscaled
// differently from the crop the model saw; otherwise it shows as a patch.
export function pasteFromTensor(img, m, tensor, size, strength) {
  const { width: w, height: h, data } = img;
  const inv = invertTransform(m);
  const corners = [[0, 0], [size, 0], [0, size], [size, size]].map(([u, v]) => applyTransform(inv, u, v));
  const xs = corners.map((c) => c[0]), ys = corners.map((c) => c[1]);
  const xa = Math.max(0, Math.floor(Math.min(...xs))), xb = Math.min(w - 1, Math.ceil(Math.max(...xs)));
  const ya = Math.max(0, Math.floor(Math.min(...ys))), yb = Math.min(h - 1, Math.ceil(Math.max(...ys)));
  const inside = (u, v) => u >= 0 && v >= 0 && u <= size - 1 && v <= size - 1;

  const shift = [0, 0, 0];
  const sumImg = [0, 0, 0], sumFace = [0, 0, 0];
  let n = 0;
  for (let y = ya; y <= yb; y += 2) {
    for (let x = xa; x <= xb; x += 2) {
      const [u, v] = applyTransform(m, x, y);
      if (!inside(u, v) || faceMask(u, v, size) < 0.9) continue;
      const p = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        sumImg[c] += data[p + c];
        sumFace[c] += sampleTensor(tensor, size, u, v, c);
      }
      n++;
    }
  }
  if (n) for (let c = 0; c < 3; c++) shift[c] = (sumImg[c] - sumFace[c]) / n;

  for (let y = ya; y <= yb; y++) {
    for (let x = xa; x <= xb; x++) {
      const [u, v] = applyTransform(m, x, y);
      if (!inside(u, v)) continue;
      const wgt = faceMask(u, v, size) * strength;
      if (wgt <= 0) continue;
      const p = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        const value = sampleTensor(tensor, size, u, v, c) + shift[c];
        data[p + c] = data[p + c] + (value - data[p + c]) * wgt;
      }
    }
  }
}

// Edge width (see analyze.js) above which a face looks soft; the same bar as
// "slightly soft" for whole images. Measured on the eyes, nose and mouth of an
// aligned crop, so a blurred background does not count.
const SOFT_EDGE_WIDTH = 4.9;
export function cropLooksSoft(tensor, size) {
  const plane = size * size;
  const x0 = Math.round(size * 0.25), y0 = Math.round(size * 0.31);
  const w = Math.round(size * 0.5), h = Math.round(size * 0.56);
  const L = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y0 + y) * size + x0 + x;
      L[y * w + x] = (0.299 * tensor[i] + 0.587 * tensor[plane + i] + 0.114 * tensor[2 * plane + i] + 1) * 127.5;
    }
  }
  const width = edgeWidth(L, w, h);
  return width == null || width > SOFT_EDGE_WIDTH;
}

// How far five detected landmarks are from human proportions: the RMS distance
// to the FFHQ template after the best similarity fit, in template eye
// distances. Real faces, even blurry or turned, stay low; animal faces and
// face-like objects (a car's front) do not.
export function templateFit(landmarks) {
  const m = similarityTransform(landmarks, FFHQ_512);
  let sum = 0;
  landmarks.forEach(([x, y], i) => {
    const [u, v] = applyTransform(m, x, y);
    sum += (u - FFHQ_512[i][0]) ** 2 + (v - FFHQ_512[i][1]) ** 2;
  });
  return Math.sqrt(sum / landmarks.length) / (FFHQ_512[1][0] - FFHQ_512[0][0]);
}

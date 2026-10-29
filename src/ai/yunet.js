// YuNet face detector (OpenCV Zoo, 2023mar): input letterboxing and output
// decoding, mirroring OpenCV's FaceDetectorYN. Pure functions, no runtime.

import { resize } from '../core/resize.js';

export const YUNET_SIZE = 640;
const STRIDES = [8, 16, 32];

// Fits img into the 640x640 input (top-left, zero padded) as a CHW float
// tensor in BGR order with raw 0..255 values, like cv2.dnn.blobFromImage.
export function yunetInput(img) {
  const scale = Math.min(YUNET_SIZE / img.width, YUNET_SIZE / img.height);
  const sw = Math.max(1, Math.round(img.width * scale));
  const sh = Math.max(1, Math.round(img.height * scale));
  const small = sw === img.width && sh === img.height ? img : resize(img, sw, sh);
  const plane = YUNET_SIZE * YUNET_SIZE;
  const data = new Float32Array(3 * plane);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const p = (y * sw + x) * 4;
      const i = y * YUNET_SIZE + x;
      data[i] = small.data[p + 2];
      data[plane + i] = small.data[p + 1];
      data[2 * plane + i] = small.data[p];
    }
  }
  return { data, scale: sw / img.width };
}

const iou = (a, b) => {
  const x1 = Math.max(a.box[0], b.box[0]), y1 = Math.max(a.box[1], b.box[1]);
  const x2 = Math.min(a.box[0] + a.box[2], b.box[0] + b.box[2]);
  const y2 = Math.min(a.box[1] + a.box[3], b.box[1] + b.box[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.box[2] * a.box[3] + b.box[2] * b.box[3] - inter || 1);
};

export function nms(faces, threshold) {
  const kept = [];
  for (const f of [...faces].sort((a, b) => b.score - a.score)) {
    if (kept.every((k) => iou(k, f) <= threshold)) kept.push(f);
  }
  return kept;
}

// outputs: { cls_8, obj_8, bbox_8, kps_8, ... } as Float32Arrays. Returns
// faces in input-tensor pixels: { score, box: [x, y, w, h], landmarks: 5x[x, y] }
// with landmarks ordered left eye, right eye, nose, left/right mouth (image order).
export function decodeYunet(outputs, { scoreThreshold = 0.6, nmsThreshold = 0.3 } = {}) {
  const faces = [];
  for (const stride of STRIDES) {
    const cols = YUNET_SIZE / stride;
    const cls = outputs[`cls_${stride}`], obj = outputs[`obj_${stride}`];
    const bbox = outputs[`bbox_${stride}`], kps = outputs[`kps_${stride}`];
    for (let idx = 0; idx < cls.length; idx++) {
      const score = Math.sqrt(Math.min(1, Math.max(0, cls[idx])) * Math.min(1, Math.max(0, obj[idx])));
      if (score < scoreThreshold) continue;
      const r = Math.floor(idx / cols), c = idx % cols;
      const cx = (c + bbox[idx * 4]) * stride, cy = (r + bbox[idx * 4 + 1]) * stride;
      const w = Math.exp(bbox[idx * 4 + 2]) * stride, h = Math.exp(bbox[idx * 4 + 3]) * stride;
      const landmarks = [];
      for (let n = 0; n < 5; n++) {
        landmarks.push([(kps[idx * 10 + 2 * n] + c) * stride, (kps[idx * 10 + 2 * n + 1] + r) * stride]);
      }
      faces.push({ score, box: [cx - w / 2, cy - h / 2, w, h], landmarks });
    }
  }
  return nms(faces, nmsThreshold);
}

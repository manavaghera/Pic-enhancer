// Tells animal faces from human ones before face restoration: the face
// detector rates a dog's face as highly as a person's, and CodeFormer would
// paste a human face onto it. A small ImageNet classifier (MobileNetV4,
// public/models/mobilenetv4) looks at the area around each face. ImageNet's
// first 398 classes are animals (fish to mammals); the rest are objects and
// scenes, which is where human heads land (wig, mask, sunglasses, jersey...).

import { resize } from '../core/resize.js';

export const CLASSIFIER_PATH = 'models/mobilenetv4/model.onnx';
export const CLASSIFIER_SIZE = 224;
const ANIMAL_CLASSES = 398;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// The square around a face box (scale times its longer side, so ears and a
// snout are in view), resized to the classifier's input as a normalised CHW
// tensor. Parts outside the picture repeat its edge.
export function classifierInput(img, [bx, by, bw, bh], scale = 1.8) {
  const side = Math.max(2, Math.round(Math.max(bw, bh) * scale));
  const x0 = Math.round(bx + bw / 2 - side / 2);
  const y0 = Math.round(by + bh / 2 - side / 2);
  const crop = { width: side, height: side, data: new Uint8ClampedArray(side * side * 4) };
  for (let y = 0; y < side; y++) {
    const row = Math.min(img.height - 1, Math.max(0, y0 + y)) * img.width;
    for (let x = 0; x < side; x++) {
      const p = (row + Math.min(img.width - 1, Math.max(0, x0 + x))) * 4;
      crop.data.set(img.data.subarray(p, p + 4), (y * side + x) * 4);
    }
  }
  const small = resize(crop, CLASSIFIER_SIZE, CLASSIFIER_SIZE);
  const plane = CLASSIFIER_SIZE * CLASSIFIER_SIZE;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) out[c * plane + i] = (small.data[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  return out;
}

// Share of the classifier's belief (softmax over its logits) on animal classes.
export function animalShare(logits) {
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  let total = 0, animal = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    total += e;
    if (i < ANIMAL_CLASSES) animal += e;
  }
  return animal / total;
}

// Face restoration with GFPGAN v1.4 or CodeFormer (ONNX Runtime Web, WebGPU
// with WASM fallback). YuNet finds faces and their landmarks; each face is
// aligned to the FFHQ template, restored at 512x512 and blended back. Loaded
// on demand inside the worker. The chosen model is fetched from Hugging Face
// on first use (see onnx.js).

import { animalShare, CLASSIFIER_PATH, CLASSIFIER_SIZE, classifierInput } from './animal.js';
import { cropLooksSoft, cropToTensor, FFHQ_512, pasteFromTensor, similarityTransform } from './faceGeometry.js';
import { FACE_MODELS } from './faceModels.js';
import { createSession, downloadVerified, ort } from './onnx.js';
import { remoteSession } from './remote.js';
import { decodeYunet, YUNET_SIZE, yunetInput } from './yunet.js';

const YUNET_PATH = 'models/yunet/face_detection_yunet_2023mar.onnx';
const SIZE = 512;
// Eye distance in the source, in pixels. Below the minimum a face is too small
// to restore. From the maximum on it is about the size the model works at (the
// template's eyes are 126 px apart), so it is only redrawn if it looks soft.
const MIN_EYE_DISTANCE = 8;
const MAX_EYE_DISTANCE = 100;

let detector = null;
// Only one restoration model is kept loaded: { id, session, backend }.
let restorer = null;
// The animal check runs on the CPU (WASM): it is small, and the GPU may be
// busy with the local AI server.
let classifier = null;

function applyPatch(bytes, patch) {
  if (!patch) return;
  const { offset, expect, value } = patch;
  if (bytes[offset - 1] !== expect[0] || bytes[offset] !== expect[1]) throw new Error('Unexpected face model layout.');
  bytes[offset] = value;
}

// Frees the GPU memory held by the face models (they load again when next needed).
export async function releaseFaces() {
  await restorer?.session.release();
  await detector?.release();
  restorer = null;
  detector = null;
}

async function ensureDetector() {
  if (!detector) {
    const res = await fetch(new URL(`${import.meta.env.BASE_URL}${YUNET_PATH}`, self.location.href));
    if (!res.ok) throw new Error('Could not load the face detector.');
    detector = (await createSession(new Uint8Array(await res.arrayBuffer()))).session;
  }
  return detector;
}

async function ensureClassifier() {
  if (!classifier) {
    const res = await fetch(new URL(`${import.meta.env.BASE_URL}${CLASSIFIER_PATH}`, self.location.href));
    if (!res.ok) throw new Error('Could not load the animal check.');
    classifier = (await createSession(new Uint8Array(await res.arrayBuffer()), ['wasm'])).session;
  }
  return classifier;
}

async function animalScore(img, box, scale) {
  const session = await ensureClassifier();
  const input = new ort.Tensor('float32', classifierInput(img, box, scale), [1, 3, CLASSIFIER_SIZE, CLASSIFIER_SIZE]);
  const out = await session.run({ [session.inputNames[0]]: input });
  return animalShare(out[session.outputNames[0]].data);
}

// An animal face: clearly one on its own, or likely one in a picture that is
// of an animal as a whole. Tuned so a dog's face (0.99) is caught while a
// real but half-hidden human face (0.58, in a picture scoring 0.13) is not.
async function isAnimal(img, box, whole) {
  const face = await animalScore(img, box, 1.8);
  if (face >= 0.8) return true;
  return face >= 0.4 && (await whole()) >= 0.5;
}

async function ensureRestorer(id, onProgress, cachedOnly) {
  const spec = FACE_MODELS[id];
  if (!spec) throw new Error(`Unknown face model: ${id}`);
  if (restorer?.id !== id) {
    await restorer?.session.release();
    restorer = null;
    const bytes = await downloadVerified(spec, (f) => onProgress?.(f, `Downloading ${spec.name} (${spec.size}, once)`), { cachedOnly });
    onProgress?.(1, `Loading ${spec.name}`);
    applyPatch(bytes, spec.patch);
    restorer = { id, ...(await createSession(bytes)) };
  }
  return restorer;
}

export async function detectFaces(img, session) {
  session ??= await ensureDetector();
  const { data, scale } = yunetInput(img);
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', data, [1, 3, YUNET_SIZE, YUNET_SIZE]) });
  const raw = Object.fromEntries(Object.entries(out).map(([k, t]) => [k, t.data]));
  return decodeYunet(raw).map((f) => ({
    score: f.score,
    box: f.box.map((v) => v / scale),
    landmarks: f.landmarks.map(([x, y]) => [x / scale, y / scale]),
  }));
}

async function restoreCrop(tensor, { session, id }) {
  const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, SIZE, SIZE]) };
  const { fidelity } = FACE_MODELS[id];
  if (fidelity != null) feeds.weight = new ort.Tensor('float32', new Float32Array([fidelity]), []);
  const out = await session.run(feeds);
  const result = out[session.outputNames[0]].data;
  for (let i = 0; i < result.length; i += 997) {
    if (!Number.isFinite(result[i])) throw new Error('The face model produced invalid output on this device.');
  }
  return result;
}

// Restores every small or soft human face in img (animal faces are left
// alone); strength 0..1 blends with the input. Faces are found in img but cropped from reference (the same picture
// at its original size, before upscaling), which is what these models are
// trained on. The large restoration model is only fetched once a face needs
// it (cachedOnly: see downloadVerified); remote runs both models on the local
// AI server instead. Returns { image, faces } where faces is the number restored.
export async function restoreFaces(img, strength, onProgress, reference = img, id = 'gfpgan', { cachedOnly = false, remote = false } = {}) {
  onProgress?.(0, 'Finding faces');
  const kx = reference.width / img.width, ky = reference.height / img.height;
  const faces = (await detectFaces(img, remote ? remoteSession('yunet') : undefined))
    .map((f) => {
      const ref = f.landmarks.map(([x, y]) => [x * kx, y * ky]);
      return { ...f, ref, eyes: Math.hypot(ref[1][0] - ref[0][0], ref[1][1] - ref[0][1]) };
    })
    .filter((f) => f.eyes >= MIN_EYE_DISTANCE)
    .sort((a, b) => a.box[2] * a.box[3] - b.box[2] * b.box[3]);
  let out = null;
  let restored = 0;
  let wholeShare = null;
  const whole = async () => (wholeShare ??= await animalScore(reference, [0, 0, reference.width, reference.height], 1));
  for (let i = 0; i < faces.length; i++) {
    const at = 0.05 + (0.95 * i) / faces.length;
    onProgress?.(at, `Restoring faces (${i + 1}/${faces.length})`);
    const [bx, by, bw, bh] = faces[i].box;
    if (await isAnimal(reference, [bx * kx, by * ky, bw * kx, bh * ky], whole)) continue;
    const crop = cropToTensor(reference, similarityTransform(faces[i].ref, FFHQ_512), SIZE);
    if (faces[i].eyes >= MAX_EYE_DISTANCE && !cropLooksSoft(crop, SIZE)) continue;
    const model = remote
      ? { id, session: remoteSession(id) }
      : await ensureRestorer(id, (f, label) => onProgress?.(at, `${label} ${Math.round(f * 100)}%`), cachedOnly);
    out ??= { width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) };
    pasteFromTensor(out, similarityTransform(faces[i].landmarks, FFHQ_512), await restoreCrop(crop, model), SIZE, strength);
    restored++;
  }
  onProgress?.(1, restored ? 'Faces restored' : 'No faces needed restoring');
  return { image: out ?? img, faces: restored };
}

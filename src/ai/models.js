// AI model catalogue. Kept free of TensorFlow.js so the UI can import it.
// Models with a url are ONNX files fetched from Hugging Face on first use and
// pinned by SHA-256; the others are bundled TensorFlow.js graph models.

export const AI_MODELS = {
  photo: {
    label: 'Photo',
    name: 'Real-ESRGAN general v3',
    blurb: 'Fast neural upscaler for photos.',
    size: '2.4 MB',
    path: 'models/realesrgan/general_fast-64/model.json',
  },
  hq: {
    label: 'Photo HQ',
    name: 'Real-ESRGAN x4plus',
    blurb: 'Rebuilds sharp fine detail and texture.',
    size: '36 MB',
    license: 'BSD-3-Clause',
    url: 'https://huggingface.co/facefusion/models-3.0.0/resolve/main/real_esrgan_x4_fp16.onnx',
    sha256: '223c117884ed92526ba1f89fbc09c663f15786c1049802c1a0ac8895b09a9cbc',
    bytes: 36144084,
    tile: 128,
    // About 30x the work of the fast models per pixel, so larger inputs are
    // scaled down first to keep a run under a minute on a typical GPU.
    maxInputPixels: 1_000_000,
    // Trained on heavily degraded photos, so it does best on blurry pictures
    // shrunk to their real detail level (aiPrescale). The fast models turn
    // such shrunk copies crunchy, so they always get the full size.
    prescale: true,
  },
  anime: {
    label: 'Art & anime',
    name: 'Real-ESRGAN AnimeVideo v3',
    blurb: 'Keeps lines and flat colors clean.',
    size: '1.2 MB',
    path: 'models/realesrgan/anime_fast-64/model.json',
  },
};

// All models upscale 4x.
export const AI_SCALE = 4;

// The optional local AI server (server/app.py): Stable Diffusion with
// ControlNet Tile on this PC's GPU, which redraws realistic detail.
export const LOCAL_AI_URL = 'http://127.0.0.1:7861';
// How far it may redraw (0..1); higher invents more and keeps less.
// Low-resolution photos (AI-upscaled) get a bold redraw, large photos, which
// already hold real detail, a gentle one.
export const LOCAL_AI_STRENGTH = 0.45;
export const LOCAL_AI_STRENGTH_LARGE = 0.3;
// It draws its detail at the output size, so nothing is stretched (and
// softened) afterwards, up to this long side; larger outputs are fitted back.
export const LOCAL_AI_MAX_SIDE = 4096;

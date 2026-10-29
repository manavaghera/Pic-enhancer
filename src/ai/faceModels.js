// Face-restoration model catalogue. Kept free of ONNX Runtime so the UI can
// import it. Both are 512x512 FFHQ-aligned models from facefusion's mirror,
// pinned by SHA-256.

export const FACE_MODELS = {
  gfpgan: {
    label: 'GFPGAN',
    name: 'GFPGAN v1.4',
    size: '340 MB',
    license: 'Apache-2.0',
    url: 'https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx',
    sha256: 'accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385',
    bytes: 340299087,
  },
  codeformer: {
    label: 'CodeFormer',
    name: 'CodeFormer',
    size: '377 MB',
    license: 'S-Lab License 1.0, non-commercial use only',
    url: 'https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx',
    sha256: '21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f',
    bytes: 376951650,
    // Balance between a cleaner face (0) and staying true to the input (1).
    fidelity: 0.5,
    // ONNX Runtime Web has no float64 Cast, so the scalar "weight" input is
    // re-declared as float32 (elem_type 11 -> 1). The graph only ever casts it
    // to float32, so behaviour is unchanged. Offset is for the pinned file.
    patch: { offset: 376834156, expect: [0x08, 0x0b], value: 0x01 },
  },
};

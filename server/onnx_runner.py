"""The web app's own ONNX models (upscaler, face finder, face fixer), run here
on CUDA while the server is up, so the browser never uses the GPU at the same
time as Stable Diffusion: Chrome's WebGPU and CUDA sharing one GPU crashed the
graphics driver. The app still does all pre- and post-processing and sends
raw float32 tensors, so results match the in-browser path exactly.

Import torch before this module: its CUDA and cuDNN libraries are what ONNX
Runtime's CUDA provider loads.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

APP = Path(__file__).resolve().parent.parent


def model_files(models):
    """name -> (file, byte patch or None); names match the app's model ids."""
    return {
        "hq": (models / "onnx" / "real_esrgan_x4_fp16.onnx", None),
        # The browser sends CodeFormer's "weight" input as float32, so the same
        # one-byte patch as in src/ai/faceModels.js is applied.
        "codeformer": (models / "onnx" / "codeformer.onnx", (376834156, (0x08, 0x0B), 0x01)),
        "yunet": (APP / "public" / "models" / "yunet" / "face_detection_yunet_2023mar.onnx", None),
    }


class OnnxModels:
    def __init__(self, models, gpu_mem_gb):
        self.files = model_files(models)
        self.sessions = {}
        cuda = {"gpu_mem_limit": int(gpu_mem_gb * 1e9), "arena_extend_strategy": "kSameAsRequested"}
        self.providers = [("CUDAExecutionProvider", cuda), "CPUExecutionProvider"]

    def available(self):
        return sorted(name for name, (path, _) in self.files.items() if path.exists())

    def session(self, name):
        if name not in self.sessions:
            path, patch = self.files[name]
            data = bytearray(path.read_bytes())
            if patch:
                offset, expect, value = patch
                if tuple(data[offset - 1 : offset + 1]) != expect:
                    raise RuntimeError(f"Unexpected layout in {path.name}.")
                data[offset] = value
            self.sessions[name] = ort.InferenceSession(bytes(data), providers=self.providers)
        return self.sessions[name]

    def run(self, name, x, weight=None):
        """Returns [(output name, float32 array)] in the model's output order."""
        s = self.session(name)
        feeds = {s.get_inputs()[0].name: x}
        if weight is not None:
            feeds["weight"] = np.array(weight, dtype=np.float32)
        outs = s.run(None, feeds)
        return [(o.name, np.ascontiguousarray(v, dtype=np.float32)) for o, v in zip(s.get_outputs(), outs)]

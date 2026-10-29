"""PixelLift local AI server.

Adds realistic detail to an already-upscaled photo with Stable Diffusion 1.5
(Realistic Vision 5.1) guided by ControlNet Tile, the technique behind
open-source "creative upscalers". While it runs it also runs the app's own
upscaler and face models (onnx_runner.py), so all AI work happens here, one job
at a time, and the browser stays off the GPU. It only listens on 127.0.0.1; the
web app uses it automatically while it runs.

Run with the environment that has its packages (see requirements.txt):
    D:/PixelLift/ai/venv/Scripts/python.exe server/app.py
Model files are read from PIXELLIFT_MODELS (default D:/PixelLift/ai/models);
server/fetch_models.py downloads them.
"""

import io
import json
import logging
import os
import socket
import threading
import urllib.request
from pathlib import Path

MODELS = Path(os.environ.get("PIXELLIFT_MODELS", "D:/PixelLift/ai/models"))
PORT = int(os.environ.get("PIXELLIFT_PORT", "7861"))


def startup_check():
    """Stops early, before the slow model load, when the port is taken."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            if json.load(r).get("ok"):
                print(f"PixelLift local AI is already running on http://127.0.0.1:{PORT}. Nothing to do.")
                raise SystemExit(0)
    except (OSError, ValueError):
        pass
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", PORT))
        except OSError:
            raise SystemExit(f"Port {PORT} is used by another program. Close that program and try again.")


if __name__ == "__main__":
    startup_check()

# Hide harmless library notices (no torchvision, no Windows log redirects,
# fp16 casting) so real errors stand out.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import numpy as np  # noqa: E402  (imported after the quick start-up check)
import torch  # noqa: E402
import uvicorn  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKL,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
)
from diffusers.utils import logging as diffusers_logging  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from onnx_runner import OnnxModels  # noqa: E402  (after torch: uses its CUDA libraries)

diffusers_logging.disable_progress_bar()

# Stable Diffusion 1.5 works best around 512-768 px, so big pictures are
# processed in overlapping tiles and blended.
TILE = 768
OVERLAP = 128
MAX_SIDE = 4096
MAX_BYTES = 64 * 1024 * 1024
# GPU safety. The browser's own AI shares this GPU, and running out of
# graphics memory can crash the whole graphics driver (and Windows with it).
# So this process is capped, checks for headroom before each job and declines
# the job, rather than risk it, when the GPU is short.
GPU_CAP_GB = float(os.environ.get("PIXELLIFT_GPU_CAP_GB", "6"))
GPU_NEEDED_GB = 2.0  # what one 768 px tile needs on top of the loaded models
ONNX_GPU_GB = 2.0  # cap for the upscaler and face models
MAX_TENSOR = 64 * 1024 * 1024  # floats in one /run request
# Worded for any subject (people, animals, cars, nature...): naming one, such
# as skin or hair, pulls everything in the picture towards it.
PROMPT = "a sharp, high quality photograph, in focus, fine natural detail, realistic textures, 35mm film grain"
NEGATIVE = (
    "blurry, soft focus, painting, illustration, cartoon, 3d render, airbrushed, oversaturated, "
    "deformed, distorted, lowres, jpeg artifacts, watermark, text"
)


def load_pipeline():
    controlnet = ControlNetModel.from_single_file(
        MODELS / "controlnet-tile" / "control_v11f1e_sd15_tile_fp16.safetensors",
        config=str(MODELS / "controlnet-tile"),
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(MODELS / "sd-vae-ft-mse", torch_dtype=torch.float16, local_files_only=True)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
        MODELS / "rv51" / "Realistic_Vision_V5.1_fp16-no-ema.safetensors",
        config=str(MODELS / "rv51-config"),
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    # The checkpoint's config names DEIS; DPM++ with Karras sigmas is sharper.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def tile_starts(length):
    """Tile origins along one axis; the last tile sits flush with the far edge."""
    size = min(TILE, length)
    if length <= size:
        return [0]
    step = size - OVERLAP
    starts = list(range(0, length - size, step))
    return starts + [length - size]


def feather(w, h, x0, y0, width, height):
    """Blend weights: linear ramps over the overlap on sides that touch a neighbour."""
    def ramp(n, lead, trail):
        r = np.ones(n, np.float32)
        k = np.linspace(0, 1, OVERLAP + 2, dtype=np.float32)[1:-1]
        if lead:
            r[:OVERLAP] = np.minimum(r[:OVERLAP], k)
        if trail:
            r[-OVERLAP:] = np.minimum(r[-OVERLAP:], k[::-1])
        return r

    rx = ramp(w, x0 > 0, x0 + w < width)
    ry = ramp(h, y0 > 0, y0 + h < height)
    return (ry[:, None] * rx[None, :])[..., None]


def color_fix(result, source, radius=24):
    """Keeps the source's colours and broad shading (anything wider than about
    radius px); the rest comes from Stable Diffusion, which otherwise drifts in
    tone tile by tile. Small radii keep too much of the blurry input."""
    r = np.asarray(result, np.float32)
    blur = lambda im: np.asarray(im.filter(ImageFilter.GaussianBlur(radius)), np.float32)
    return r - blur(result) + blur(source)


class Refiner:
    def __init__(self):
        self.pipe = load_pipeline()
        self.lock = threading.Lock()
        self.progress = {"done": 0, "total": 0}

    # Tuned on a blurry band photo against Artguru: 0.45 strength with these
    # settings gives real hair, skin and fabric without changing who people are
    # too much (stronger redraws start inventing new faces).
    def refine(self, img, strength, steps=24, guidance=5.0, control=0.7, fix=24, seed=7):
        w, h = img.size
        # Stable Diffusion needs sides that are multiples of 8.
        pw, ph = -(-w // 8) * 8, -(-h // 8) * 8
        padded = Image.new("RGB", (pw, ph))
        padded.paste(img, (0, 0))
        if pw > w:
            padded.paste(img.crop((w - 1, 0, w, h)).resize((pw - w, h)), (w, 0))
        if ph > h:
            padded.paste(padded.crop((0, h - 1, pw, h)).resize((pw, ph - h)), (0, h))
        xs, ys = tile_starts(pw), tile_starts(ph)
        out = np.zeros((ph, pw, 3), np.float32)
        weight = np.zeros((ph, pw, 1), np.float32)
        self.progress = {"done": 0, "total": len(xs) * len(ys)}
        for y0 in ys:
            for x0 in xs:
                tw, th = min(TILE, pw), min(TILE, ph)
                tile = padded.crop((x0, y0, x0 + tw, y0 + th))
                res = self.pipe(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE,
                    image=tile,
                    control_image=tile,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    controlnet_conditioning_scale=control,
                    width=tw,
                    height=th,
                    generator=torch.Generator("cuda").manual_seed(seed),
                ).images[0]
                m = feather(tw, th, x0, y0, pw, ph)
                out[y0 : y0 + th, x0 : x0 + tw] += color_fix(res, tile, fix) * m
                weight[y0 : y0 + th, x0 : x0 + tw] += m
                self.progress["done"] += 1
        merged = (out / np.maximum(weight, 1e-6)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(merged).crop((0, 0, w, h))


refiner = None
onnx_models = None
app = FastAPI(title="PixelLift local AI")
# Only pages served from this computer may call it.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Outputs"],
)


@app.get("/")
def index():
    return Response(
        "PixelLift local AI is running. Keep this window open and use the app at http://localhost:5179.\n",
        media_type="text/plain",
    )


@app.get("/health")
def health():
    return {
        "ok": True,
        "gpu": torch.cuda.get_device_name(0),
        "model": "Realistic Vision 5.1 + ControlNet Tile",
        "onnx": onnx_models.available(),
    }


@app.post("/run/{model}")
async def run_model(model: str, request: Request, shape: str, weight: float | None = None):
    """Runs one of the app's ONNX models on a raw float32 tensor (the request
    body, in the given NCHW shape). The response body holds the outputs back to
    back; X-Outputs lists their names and shapes."""
    if model not in onnx_models.available():
        raise HTTPException(404, f"Model {model} is not on this server.")
    try:
        dims = [int(v) for v in shape.split(",")]
    except ValueError:
        raise HTTPException(400, "shape must be four comma-separated integers.")
    count = int(np.prod(dims)) if dims else 0
    if len(dims) != 4 or min(dims) < 1 or count > MAX_TENSOR:
        raise HTTPException(400, "shape must be four positive sizes within the size limit.")
    body = await request.body()
    if len(body) != count * 4:
        raise HTTPException(400, "The body does not match the shape.")
    x = np.frombuffer(body, np.float32).reshape(dims)

    def run():
        with refiner.lock:
            return onnx_models.run(model, x, weight)

    try:
        outs = await run_in_threadpool(run)
    except Exception as err:
        raise HTTPException(500, f"{model} failed: {err}")
    header = json.dumps([{"name": name, "shape": list(v.shape)} for name, v in outs])
    return Response(b"".join(v.tobytes() for _, v in outs), media_type="application/octet-stream", headers={"X-Outputs": header})


@app.get("/progress")
def progress():
    return refiner.progress


@app.post("/refine")
async def refine(request: Request, strength: float = 0.45):
    body = await request.body()
    if not body or len(body) > MAX_BYTES:
        raise HTTPException(413, "Send one PNG or JPEG image under 64 MB.")
    try:
        img = Image.open(io.BytesIO(body))
        img.load()
    except Exception:
        raise HTTPException(400, "The request body is not a readable image.")
    if max(img.size) > MAX_SIDE or min(img.size) < 64:
        raise HTTPException(400, f"Image sides must be between 64 and {MAX_SIDE} px.")
    strength = min(0.6, max(0.1, strength))

    def run():
        with refiner.lock:
            room = gpu_headroom_gb()
            if room < GPU_NEEDED_GB:
                raise HTTPException(503, f"The GPU is busy right now ({room:.1f} GB free), so this step was skipped.")
            try:
                return refiner.refine(img.convert("RGB"), strength)
            except torch.cuda.OutOfMemoryError:
                raise HTTPException(503, "The GPU ran out of memory, so this step was skipped.")
            finally:
                # Hand unused memory back so the browser's AI can use it.
                torch.cuda.empty_cache()

    result = await run_in_threadpool(run)
    buf = io.BytesIO()
    result.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


def gpu_headroom_gb():
    """Memory a job could still get: free on the device (plus what this process
    holds unused), within this process's cap."""
    free, _ = torch.cuda.mem_get_info()
    held_unused = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    under_cap = GPU_CAP_GB * 1e9 - torch.cuda.memory_allocated()
    return min(free + held_unused, under_cap) / 1e9


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found; the local AI server needs an NVIDIA GPU.")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.9, GPU_CAP_GB * 1e9 / total))
    print(f"Loading Stable Diffusion models (GPU memory capped at {GPU_CAP_GB:g} GB)...", flush=True)
    refiner = Refiner()
    onnx_models = OnnxModels(MODELS, ONNX_GPU_GB)
    print(f"PixelLift local AI ready on http://127.0.0.1:{PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

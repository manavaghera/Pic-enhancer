"""Downloads the local AI server's models (about 3.6 GB) and verifies every
large file against the SHA-256 Hugging Face publishes for it. Safe to re-run:
files already downloaded and intact are skipped.

    D:/PixelLift/ai/venv/Scripts/python.exe server/fetch_models.py [models folder]

The folder defaults to PIXELLIFT_MODELS or D:/PixelLift/ai/models.
"""
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PIXELLIFT_MODELS", "D:/PixelLift/ai/models"))
FILES = [
    # (repo, file in repo, local path)
    ("SG161222/Realistic_Vision_V5.1_noVAE", "Realistic_Vision_V5.1_fp16-no-ema.safetensors", "rv51/Realistic_Vision_V5.1_fp16-no-ema.safetensors"),
    ("SG161222/Realistic_Vision_V5.1_noVAE", "model_index.json", "rv51-config/model_index.json"),
    ("SG161222/Realistic_Vision_V5.1_noVAE", "scheduler/scheduler_config.json", "rv51-config/scheduler/scheduler_config.json"),
    ("SG161222/Realistic_Vision_V5.1_noVAE", "text_encoder/config.json", "rv51-config/text_encoder/config.json"),
    ("SG161222/Realistic_Vision_V5.1_noVAE", "unet/config.json", "rv51-config/unet/config.json"),
    ("SG161222/Realistic_Vision_V5.1_noVAE", "vae/config.json", "rv51-config/vae/config.json"),
    ("SG161222/Realistic_Vision_V5.1_noVAE", "tokenizer/vocab.json", "rv51-config/tokenizer/vocab.json"),
    ("SG161222/Realistic_Vision_V5.1_noVAE", "tokenizer/merges.txt", "rv51-config/tokenizer/merges.txt"),
    ("SG161222/Realistic_Vision_V5.1_noVAE", "tokenizer/special_tokens_map.json", "rv51-config/tokenizer/special_tokens_map.json"),
    ("SG161222/Realistic_Vision_V5.1_noVAE", "tokenizer/tokenizer_config.json", "rv51-config/tokenizer/tokenizer_config.json"),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.safetensors", "sd-vae-ft-mse/diffusion_pytorch_model.safetensors"),
    ("stabilityai/sd-vae-ft-mse", "config.json", "sd-vae-ft-mse/config.json"),
    ("comfyanonymous/ControlNet-v1-1_fp16_safetensors", "control_v11f1e_sd15_tile_fp16.safetensors", "controlnet-tile/control_v11f1e_sd15_tile_fp16.safetensors"),
    ("lllyasviel/control_v11f1e_sd15_tile", "config.json", "controlnet-tile/config.json"),
    # The same upscaler and face model the browser uses, run here on CUDA.
    ("facefusion/models-3.0.0", "real_esrgan_x4_fp16.onnx", "onnx/real_esrgan_x4_fp16.onnx"),
    ("facefusion/models-3.0.0", "codeformer.onnx", "onnx/codeformer.onnx"),
]


def lfs_hashes(repo):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true") as r:
        info = json.load(r)
    return {s["rfilename"]: s.get("lfs", {}).get("sha256") for s in info.get("siblings", [])}


def main():
    hashes = {}
    for repo, name, local in FILES:
        if repo not in hashes:
            hashes[repo] = lfs_hashes(repo)
        dest = ROOT / local
        expected = hashes[repo].get(name)
        if dest.exists() and expected and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
            print(f"ok (cached) {local}", flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        digest = hashlib.sha256()
        with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/{name}") as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got, last = 0, 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                digest.update(chunk)
                got += len(chunk)
                if total and got - last > total / 10:
                    last = got
                    print(f"  {local}: {got / total:.0%}", flush=True)
        if expected and digest.hexdigest() != expected:
            tmp.unlink()
            sys.exit(f"SHA-256 mismatch for {local}; file discarded")
        tmp.replace(dest)
        print(f"ok {local} ({got / 1e6:.1f} MB{', verified' if expected else ''})", flush=True)


main()

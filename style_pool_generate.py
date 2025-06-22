#!/usr/bin/env python3
# coding: utf-8
"""
Erzeugt Style-Panels (HiDream + Comic-LoRA).
Speichert sie unter <out_dir>/panel_XXXXX.png
"""

import argparse
import random
from itertools import product
from pathlib import Path
from typing import Any

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from tqdm import tqdm

POSE_TOKENS = [
    "full-body, low angle",
    "medium shot, over-the-shoulder",
    "medium shot, 3/4 view",
    "close-up, eye-level",
    "close-up, dutch angle",
]
EXPRESSIONS = [
    "happy expression", "serious expression", "shouting",
    "surprised", "angry", "sad expression",
]

def main(out_dir: str, n_images: int):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    pipe : Any = DiffusionPipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Dev", torch_dtype="auto"
    ).to(device)
    #pipe.load_lora_weights("your-org/comic-lora-hidream")

    combos = list(product(POSE_TOKENS, EXPRESSIONS))
    random.shuffle(combos)

    for idx in tqdm(range(n_images), desc="render panels"):
        pose, expr = combos[idx % len(combos)]
        prompt = f"comic panel, {pose}, {expr}, fixed style"
        img = pipe(prompt=prompt,
                   num_inference_steps=28,
                   guidance_scale=5.5,
                   height=1024, width=1024).images[0]
        filename = f"panel_{idx:05d}.png"
        img.save(out_path / filename)
        print(f"generated {filename}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./style_pool")
    ap.add_argument("--n_images", type=int, default=5)
    args = ap.parse_args()
    main(args.out_dir, args.n_images)

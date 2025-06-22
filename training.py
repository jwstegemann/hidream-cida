from urllib.parse import scheme_chars
import os
from pathlib import Path

import modal

# ─────────────────────────── Config ──
APP_NAME = "flux-kohya-ss"
PORT = 7860
HF_CACHE = Path("/models")      # shared HF cache
DATA_DIR = Path("/data")        # user dataset mount
from json.decoder import scanstring

# ─────────────────────────── Modal app ──
app = modal.App(APP_NAME)

# Volumes
hf_cache_vol = modal.Volume.from_name("cida-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("cida-data", create_if_missing=True)

# Secret
HF_SECRET = modal.Secret.from_name("huggingface")
HF_CACHE = Path("/models")      # shared HF cache
DATA_DIR = Path("/data")
EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu121"
TEXT_ENCODER_4 = "unsloth/Meta-Llama-3.1-8B-Instruct"

# Container image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "git-lfs",
        "wget",
        "ca-certificates",
        "build-essential",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "ffmpeg",# base utilities
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        extra_index_url=EXTRA_INDEX_URL,
    )
    .pip_install_from_requirements(
        "./requirements.txt",
        extra_index_url=EXTRA_INDEX_URL
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",   # suppress TF INFO logs
        }
    )
)


# ────────────────── Create Comic Training Images ──

import random
from itertools import product
from pathlib import Path

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

@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={str(HF_CACHE): hf_cache_vol, str(DATA_DIR): data_vol},
    gpu="L40S",
    timeout=60 * 60,
)
def generate_style_pool():
    n_images: int = 2
    out_path = DATA_DIR / "training/style"
    out_path.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    import torch
    from diffusers import HiDreamImagePipeline
    from huggingface_hub import login
    from transformers import AutoTokenizer, LlamaForCausalLM

    login(token=hf_token)

    tokenizer_4 = AutoTokenizer.from_pretrained(
        TEXT_ENCODER_4, # "meta-llama/Meta-Llama-3.1-8B-Instruct",
        cache_dir=HF_CACHE,
        token=hf_token,
    )
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        TEXT_ENCODER_4, # "meta-llama/Meta-Llama-3.1-8B-Instruct",
        output_hidden_states=True,
        output_attentions=True,
        return_dict_in_generate=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        token=hf_token,
    )

    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0, use_dynamic_shifting=False)

    pipe = HiDreamImagePipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Dev",
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        token=hf_token,
    )
    pipe.enable_model_cpu_offload()
    #pipe.load_lora_weights("your-org/comic-lora-hidream")

    combos = list(product(POSE_TOKENS, EXPRESSIONS))
    random.shuffle(combos)

    for idx in tqdm(range(n_images), desc="render panels"):
        pose, expr = combos[idx % len(combos)]
        prompt = f"A detailed realistic illustration in graphic novel style of: a man, {pose} with {expr} expression. cinematic lighting, perfect composition. Style: simple shapes, (american comic inking:1.2), (strong inking lines:1.1), hatch shading, dynamic foreshortening, rough linework with vibrant flat colors, gritty realistic style, hand‑drawn aesthetic, all in sharp focus. Background: sharp, (rough pencil lines), minimal background elements. Face:0.8, Hair:0.8 to preserve stylized portrayal,(realistic face and expression):0.8"
        negative_prompt = "(anime face):0.8, (cartoon face):0.8, too detailed hair, photorealistic face, ultra‑detailed face, realistic hair, sharp hair strands, cluttered background, watercolor, oil painting, smooth shading, cute or exaggerated features, signature, depth of field, blurry, blur, detailed background"



        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=1024,
            width=1024,
            guidance_scale=0.0,
            num_inference_steps=28,
            generator=torch.Generator("cuda").manual_seed(0),
        ).images[0]
        filename = f"panel_{idx:05d}.png"
        image.save(out_path / filename)
        print(f"Saved {filename}")

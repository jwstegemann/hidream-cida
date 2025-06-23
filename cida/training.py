#!/usr/bin/env python3
# HiDream Comic-ID Training on Modal – Fast-Branch gegen HiDream-I1-Fast
# ---------------------------------------------------------------
# CUDA ≥ 12.1, torch 2.4, diffusers @ main, accelerate 0.34, deepspeed 0.14
# ---------------------------------------------------------------
import itertools
import random
from pathlib import Path

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------- Dataset ----------
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .generate_style_pool import (
    APP_NAME,
    DATA_DIR,
    HF_CACHE,
    HF_SECRET,
    TEXT_ENCODER_4,
    data_vol,
    hf_cache_vol,
    image,
)

app = modal.App(APP_NAME)

class ComicDataset(Dataset):
    def __init__(self, root: Path, size=1024):
        self.real  = list((root/"real").glob("*.[jp][pn]*"))
        self.style = list((root/"style").glob("*.[jp][pn]*"))
        assert self.real and self.style, "Leere ‹real› oder ‹style›-Ordner!"
        self.t = transforms.Compose([
            transforms.Resize((size,size), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)])
    def __len__(self): return max(len(self.real),len(self.style))
    def __getitem__(self, i):
        return {
            "real" : self.t(Image.open(self.real[i%len(self.real)]).convert("RGB")),
            "style": self.t(Image.open(random.choice(self.style)).convert("RGB"))
        }

# ---------- Model-Bausteine ----------
from insightface.model_zoo import get_model as get_insight


class ArcFace(nn.Module):
    """Gefrorener 512-d ArcFace-Encoder (ONNX-Backend)"""
    def __init__(self, device, name: str = "antelopev2"):
        super().__init__()
        self.model = get_insight(name, root=str(HF_CACHE) + "/.insightface", download=False)
        if self.model is None:
            raise RuntimeError(
                f"InsightFace-Gewichte '{name}' nicht gefunden. "
                "Lade sie mit:  insightface-cli model.download antelopev2"
            )
        # Session für GPU oder CPU öffnen
        self.model.prepare(ctx_id=0 if device.type == "cuda" else -1)

    @torch.no_grad()
    def forward(self, x):                    # x: [-1,1] RGB, float32/16
        # -> BGR, 0-255, H=W=112
        x = F.interpolate(x, (112, 112), mode="bilinear")
        x = ((x + 1) / 2) * 255.0
        x = x[:, [2, 1, 0]].permute(0, 2, 3, 1).cpu().numpy().astype("float32")
        emb = self.model.get(x)              # [B,512]  – InsightFace API
        return F.normalize(
            torch.from_numpy(emb).to(x.device), dim=-1
        )                                    # zurück nach Torch

class ProjectionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512,1024), nn.SiLU(), nn.Linear(1024,768))
    def forward(self,x): return self.net(x)

from timm.models.vision_transformer import Block as ViTBlock


class InfuseNet(nn.Module):
    def __init__(self, d=768,h=12,L=12):
        super().__init__()
        self.blocks = nn.ModuleList([ViTBlock(dim=d,
            num_heads=h,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm # type: ignore
        ) for _ in range(L)])
        self.norm = nn.LayerNorm(d)
    def forward(self,x):
        for blk in self.blocks: x = blk(x)
        return self.norm(x)

class ResidualHook:
    def __init__(self, mod): self.r=None; self.h=mod.register_forward_hook(self.f)
    def f(self,_,__,o): return o+self.r.unsqueeze(1) if self.r is not None else o
    def set(self,r): self.r=r
    def clr(self): self.r=None
    def rm(self):  self.h.remove()

def cos(a,b): return F.cosine_similarity(a,b).mean()

def createPipe(model="", shift=6.0):
    import os

    from diffusers.pipelines.hidream_image.pipeline_hidream_image import (
        HiDreamImagePipeline,
    )
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from huggingface_hub import login
    from transformers import AutoTokenizer, LlamaForCausalLM

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
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

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)

    pipe = HiDreamImagePipeline.from_pretrained(
        model,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        token=hf_token,
    )
    pipe.enable_model_cpu_offload()
    return pipe


@torch.no_grad()
def lfm_loss(transformer, vae, imgs_m1, text_emb, t=None):
    """
    imgs_m1  : [-1,1] RGB – z.B. Style-Panel aus Dataset
    text_emb : Text/ID Embeddings vom Prompt-Encoder (+ InfuseNet-Token)
    returns  : L_FM (scalar)
    """
    bs = imgs_m1.size(0)
    device = imgs_m1.device

    # 1) x0  – sauberer Latent
    with torch.no_grad():
        imgs_01 = (imgs_m1 + 1) / 2
        x0 = vae.encode(imgs_01).latent_dist.sample() * 0.18215   # SD-Skalierung

    # 2) x1  – reines Rauschen
    x1 = torch.randn_like(x0)

    # 3) Zufälliges t  ∈ (0,1)
    if t is None:
        t = torch.rand(bs, device=device)
    t = t.view(bs, 1, 1, 1)

    x_t   = (1 - t) * x0 + t * x1
    u_star = x1 - x0                    # Ziel-Velocity

    # 4) Modell-Prediction
    #    Das HiDream-transformer erwartet Timesteps im DDPM-Format → (t*1000).round()
    steps = (t * 1000).round().long()
    u_hat = transformer(x_t, steps, encoder_hidden_states=text_emb).sample

    # 5) L2-Loss
    return F.mse_loss(u_hat, u_star)


def collect_ff_after_blocks(model, every: int = 4):
    """
    Sammelt die FFN-Module nach Block 4, 8, …, 48 – egal ob
    - alter UNet (down_blocks + mid_block) oder
    - neuer HiDream-Transformer (double/single_stream_blocks).
    """
    targets = []

    # ▸ Neuer HiDream-Transformer
    if hasattr(model, "double_stream_blocks"):
        blocks = list(model.double_stream_blocks) + list(model.single_stream_blocks)
        for i, blk in enumerate(blocks):
            if (i + 1) % every == 0:          # 4-er Raster
                ff = getattr(blk, "ff_i", None)              # Single-/Double-Block
                if ff is None and hasattr(blk, "block"):     # Wrapper-Fallback
                    ff = getattr(blk.block, "ff_i", None) or getattr(blk.block, "ff", None)
                if ff is None:
                    raise RuntimeError(f"Kein FFN in Block {i}")
                targets.append(ff)

    # ▸ Alter UNet-Pfad (falls Du noch eines der Dev-Checkpoints nutzt)
    elif hasattr(model, "down_blocks"):
        total = 48
        for i in range(every - 1, total, every):             # 3,7,…,47
            if i < 48:
                blk = model.down_blocks[i // 4].transformer_blocks[i % 4]
            else:
                blk = model.mid_block.transformer_blocks[0]
            targets.append(blk.ff)
    else:
        raise AttributeError("Unbekannte HiDream-Struktur – weder down_blocks noch *stream_blocks gefunden")

    return targets

def models_download():
    import os

    from huggingface_hub import snapshot_download

    huggingface_token: str = os.environ["HUGGINGFACE_TOKEN"]

    local_dir = str(HF_CACHE) + "/.insightface/models" + "/" + "antelopev2"
    snapshot_download(
        repo_id="DavidHoa/antelopev2",
        local_dir= local_dir,
        token=huggingface_token,
    )

# ---------- Modal-Function ----------
@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={str(HF_CACHE):hf_cache_vol, str(DATA_DIR):data_vol},
    gpu="A100",
    timeout=60*60*8,
)
def train(steps:int=20_000,batch:int=4,warm:int=500,guidance_scale:float=0.0,train_fast:bool=False):
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from diffusers.optimization import get_cosine_schedule_with_warmup

    models_download()

    device=torch.device("cuda")

    acc=Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(
        find_unused_parameters=not train_fast)])

    # ---------- Adapters ----------
    idenc=ArcFace(device)
    proj =ProjectionMLP().to(device)
    infuse=InfuseNet().to(device)

    # ---------- Pipelines ----------
    pipe = createPipe("HiDream-ai/HiDream-I1-Dev", shift=6.0)
    # TODO: Lora aktivieren
    # pipe.load_lora_weights("your-org/comic-lora-hidream")

    pipe_fast = createPipe("HiDream-ai/HiDream-I1-Fast", shift=3.0)
    for p in pipe_fast.transformer.parameters():
        p.requires_grad_(train_fast)

    # ---------- Hooks ----------
    hooks=[ResidualHook(m) for m in collect_ff_after_blocks(pipe.transformer)]

    # ---------- Optimizer ----------
    params=itertools.chain(proj.parameters(),infuse.parameters(),
                           pipe_fast.transformer.parameters() if train_fast else [])

    from torch.optim import AdamW  # type: ignore

    opt=AdamW(params,lr=1e-4,weight_decay=1e-2)
    sch=get_cosine_schedule_with_warmup(opt,warm,steps)

    # ---------- Data ----------
    dl=DataLoader(ComicDataset(DATA_DIR / "training"),batch_size=batch,shuffle=True,
                  num_workers=4,pin_memory=True,persistent_workers=True)
    dl=acc.prepare(dl); proj,infuse,pipe_fast,opt,sch=acc.prepare(
        proj,infuse,pipe_fast,opt,sch)

    global_step=0
    for epoch in range(9999):
        for b in dl:
            with acc.accumulate(proj):
                real=b["real"].to(device)
                id_gt=idenc(real)

                # --- shared noise in latent space  [B,4,128,128] ---
                noise = torch.randn((real.size(0),4,128,128),
                                  device=device,dtype=torch.bfloat16)

                # --- Pfad A  (ID-Injection) ---
                inf_vec=infuse(proj(id_gt).unsqueeze(1)).squeeze(1)
                for h in hooks: h.set(inf_vec)
                lat_A=pipe(prompt=["portrait, color, cinematic"]*real.size(0),
                           latents=noise,guidance_scale=guidance_scale, # type: ignore
                           num_inference_steps=28,output_type="latent").images  # type: ignore
                for h in hooks:
                    h.clr()

                # --- Pfad B (Style) ---
                lat_B=pipe(prompt=["comic panel, fixed style"]*real.size(0),
                           latents=noise,guidance_scale=guidance_scale, # type: ignore
                           num_inference_steps=28,output_type="latent").images  # type: ignore

                # --- Fast-Branch x₀-Schätzung ---
                x0_lat=pipe_fast(prompt=["portrait"]*real.size(0),
                                 latents=noise,guidance_scale=guidance_scale,
                                 num_inference_steps=14,output_type="latent").images
                # decode → [-1,1] RGB
                x0_rgb=pipe.vae.decode(x0_lat/pipe.vae.config.scaling_factor,
                                       return_dict=False)[0]
                id_pred=idenc(x0_rgb)

                # ---------- Verluste ----------
                Lid = 1 - cos(id_pred,id_gt)
                Lalign_sem    = F.mse_loss(lat_A.mean(1),lat_B.mean(1)) # type: ignore
                Lalign_layout = F.mse_loss(lat_A,lat_B) # type: ignore
                Lalign=0.6*Lalign_sem + 0.1*Lalign_layout

                # --- Flow-Matching Loss ---------------------------------
                # imgs_m1: Style-Panel aus Batch (siehe Dataset)
                style_rgb = b["style"].to(device)          # [-1,1] RGB
                # Text-Embeddings für Style-Prompt (Pfad B)
                text_emb = pipe._encode_prompt(
                    ["comic panel, fixed style"] * style_rgb.size(0),
                    device=device,
                    do_classifier_free_guidance=False
                )

                Ldiff = lfm_loss(
                    pipe.transformer,        # gefrorenes HiDream-transformer
                    pipe.vae,
                    imgs_m1 = style_rgb,
                    text_emb = text_emb
                )

                loss=Ldiff + Lid + Lalign
                acc.backward(loss)
                opt.step()
                sch.step()
                opt.zero_grad()
            global_step+=1
            if acc.is_main_process and global_step%50==0:
                print(f"[{global_step}] loss={loss.item():.4f} "
                      f"Lid={Lid.item():.3f} Lalign={Lalign.item():.3f}")
            if global_step>=steps: break
        if global_step>=steps: break

    acc.wait_for_everyone()
    if acc.is_main_process:
        torch.save({
            "proj":acc.get_state_dict(proj),
            "infuse":acc.get_state_dict(infuse),
            "fast_transformer":acc.get_state_dict(pipe_fast.transformer) if train_fast else None},
            "hidream_comic_id_adapter.pth")
        print("✅ Training abgeschlossen – Adapter gespeichert.")

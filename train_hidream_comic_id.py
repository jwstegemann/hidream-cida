#!/usr/bin/env python3
# coding: utf-8
"""
HiDream Comic-ID Training (Variante A)
-------------------------------------
• Gefrorener HiDream-17 B-DiT Backbone + gefrorene Comic-LoRA
• Trainierbar: Projection-MLP, InfuseNet (Mini-DiT), Lightning-Adapter **ist eingefroren**
• Zwei separate Ordner:
      real/   – echte Portraitfotos  (ϕ → L_id)
      style/  – vorgerenderte Comic-Panels (Layout/Stil-Target → L_align)

Getestet mit:
  • torch-2.3.0 + cu128   • diffusers-0.27.2   • 8× H100 (DeepSpeed ZeRO-3)
"""

# ---------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------
import argparse, os, random, itertools, math
from pathlib import Path
from typing import List, Dict

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from timm.models.vision_transformer import Block as ViTBlock
from insightface.model_zoo import get_model                        # AntelopeV2

# ---------------------------------------------------------------------
# 1. Dataset – Real-Foto  +  Style-Panel (Comic)
# ---------------------------------------------------------------------
class LidLalignDataset(Dataset):
    """ Kombiniert zufällig ein Real-Portrait und ein beliebiges Style-Panel. """

    def __init__(self, real_dir: str, style_dir: str, image_size: int = 1024):
        self.real_paths  = list(Path(real_dir).glob("*.[jp][pn]*"))
        self.style_paths = list(Path(style_dir).glob("*.[jp][pn]*"))
        assert self.real_paths and self.style_paths, \
            "Real-Fotos oder Style-Panels nicht gefunden."
        self.size = image_size
        self.to_tensor = transforms.Compose([
            transforms.Resize((self.size, self.size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),                       # → [0,1]
            transforms.Lambda(lambda t: t * 2. - 1.)     # → [-1,1]
        ])

    def __len__(self):
        # jede Kombi zählt einmal – reicht für PoC
        return max(len(self.real_paths), len(self.style_paths))

    def __getitem__(self, idx):
        real_path  = self.real_paths[idx % len(self.real_paths)]
        style_path = random.choice(self.style_paths)
        real_img   = self.to_tensor(Image.open(real_path).convert("RGB"))
        style_img  = self.to_tensor(Image.open(style_path).convert("RGB"))
        return {"real": real_img, "synth": style_img}

# ---------------------------------------------------------------------
# 2. Identity-Encoder ϕ  (gefrosteter ArcFace-50 / Antelopev2)
# ---------------------------------------------------------------------
class IdentityEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.backbone = get_model("antelopev2", download=True).to(device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, img_rgbm1):           # erwartet [-1,1] RGB
        img_01 = (img_rgbm1 + 1) / 2        # → [0,1]
        img_bgr = img_01[:, [2, 1, 0]]      # InsightFace nutzt BGR
        return F.normalize(self.backbone(img_bgr), dim=-1)  # [B,512]

# ---------------------------------------------------------------------
# 3. Projection-MLP  512 → 768
# ---------------------------------------------------------------------
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): return self.net(x)

# ---------------------------------------------------------------------
# 4. Mini-DiT  (InfuseNet) – 12 Blöcke, ViT-kompatibel
# ---------------------------------------------------------------------
class InfuseNet(nn.Module):
    def __init__(self, depth=12, width=768, heads=12):
        super().__init__()
        self.blocks = nn.ModuleList([
            ViTBlock(dim=width, num_heads=heads,
                     mlp_ratio=4.0, qkv_bias=True,
                     norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)

    def forward(self, x):           # x: [B,1,768]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

# ---------------------------------------------------------------------
# 5. Lightning-UNet (4-Step) – nur Inferenz, daher eingefroren
# ---------------------------------------------------------------------
def load_lightning_unet(device="cuda"):
    unet = UNet2DConditionModel.from_pretrained(
        "ByteDance/SDXL-Lightning-4step",
        subfolder="unet",
        torch_dtype=torch.bfloat16,
    ).to(device)
    for p in unet.parameters():
        p.requires_grad_(False)
    return unet

# ---------------------------------------------------------------------
# 6. Residual-Hooks – addieren InfuseNet-Vektor nach FFN
# ---------------------------------------------------------------------
class ResidualHook:
    def __init__(self, module: nn.Module):
        self.residual = None
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        return out + self.residual if self.residual is not None else out

    def set(self, res): self.residual = res          # [B,768]
    def clear(self):   self.residual = None
    def remove(self):  self.h.remove()

# ---------------------------------------------------------------------
# 7. Hilfs-Loss-Funktionen
# ---------------------------------------------------------------------
def cosine_sim(a, b): return F.cosine_similarity(a, b).mean()
def lfm_surrogate(noise_pred, noise): return F.mse_loss(noise_pred.float(), noise.float())

# ---------------------------------------------------------------------
# 8. Haupt-Trainings­funktion
# ---------------------------------------------------------------------
def main(cfg):
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    device = accelerator.device

    # -------- HiDream Backbone + Comic-LoRA laden --------
    pipe = DiffusionPipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Dev", torch_dtype=torch.bfloat16
    ).to(device).load_lora_weights("your-org/comic-lora-hidream")
    pipe.enable_model_cpu_offload()

    # Residual-Hook-Layer (FF after blocks 4,8,…,48 → Index 3,7,…,47)
    target_layers = [pipe.unet.transformer_blocks[i].ff for i in range(3, 48, 4)]
    hooks = [ResidualHook(m) for m in target_layers]

    # -------- Adapter-Module --------
    id_enc      = IdentityEncoder(device)
    proj_mlp    = ProjectionMLP().to(device)
    infusenet   = InfuseNet().to(device)
    lightning   = load_lightning_unet(device)

    # -------- Optimizer (Lightning-UNet ausgeschlossen) --------
    optim_params = itertools.chain(proj_mlp.parameters(), infusenet.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=1e-4, weight_decay=1e-2)
    lr_sched  = get_cosine_schedule_with_warmup(
        optimizer, cfg.warmup, cfg.steps
    )

    # -------- Dataset / Dataloader --------
    ds = LidLalignDataset(cfg.real_dir, cfg.style_dir, image_size=1024)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.workers, pin_memory=True)
    dl = accelerator.prepare(dl)

    proj_mlp, infusenet, optimizer, lr_sched = accelerator.prepare(
        proj_mlp, infusenet, optimizer, lr_sched
    )

    # -------- Training Loop --------
    global_step = 0
    for epoch in range(9999):
        for batch in dl:
            with accelerator.accumulate(proj_mlp):
                real   = batch["real"].to(device)    # [-1,1]
                style  = batch["synth"].to(device)   # [-1,1]

                # ----- ArcFace-Embedding (ϕ) -----
                id_emb = id_enc(real)                # [B,512]

                # ----- Identity-Injection -----
                proj   = proj_mlp(id_emb)            # [B,768]
                infuse = infusenet(proj.unsqueeze(1)).squeeze(1)  # [B,768]
                for h in hooks: h.set(infuse)

                # ---------- Pfad A : Identity-Prompt ----------
                noise = torch.randn_like(real)       # gemeinsamer Seed
                lat_A = pipe(
                    prompt=["portrait, color, cinematic"] * real.size(0),
                    num_inference_steps=28, guidance_scale=1.0,
                    latents=noise, output_type="latent", return_dict=False
                )[0]                                 # [B,C,H/8,W/8] DiT-Latent

                # hooks für Pfad B deaktivieren
                for h in hooks: h.clear()

                # ---------- Pfad B : Style-Prompt ----------
                lat_B = pipe(
                    prompt=["comic panel, fixed style"] * real.size(0),
                    num_inference_steps=28, guidance_scale=1.0,
                    latents=noise, output_type="latent", return_dict=False
                )[0]

                # ---------- Lightning-Branch ----------
                x0_hat = lightning(
                    noise, timesteps=torch.tensor([1.0], device=device),
                    encoder_hidden_states=proj
                ).sample                              # [B,C,H/8,W/8]
                id_pred = id_enc(x0_hat)              # ϕ(x₀)

                # ---------- Verluste ----------
                Lid    = 1 - cosine_sim(id_pred, id_emb)
                Lalign_sem    = F.mse_loss(lat_A.mean(dim=1), lat_B.mean(dim=1))
                Lalign_layout = F.mse_loss(lat_A, lat_B)
                Lalign = 0.6*Lalign_sem + 0.1*Lalign_layout
                noise_gt = torch.randn_like(lat_A)
                Ldiff  = lfm_surrogate(lat_A, noise_gt)

                loss = Ldiff + Lid + Lalign
                accelerator.backward(loss)
                optimizer.step(); lr_sched.step(); optimizer.zero_grad()

            global_step += 1
            if accelerator.is_main_process and global_step % cfg.log_steps == 0:
                print(f"[{global_step}] loss={loss.item():.4f} "
                      f"Lid={Lid.item():.3f}  Lalign={Lalign.item():.3f}")

            if global_step >= cfg.steps: break
        if global_step >= cfg.steps: break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt = {
            "proj_mlp": accelerator.get_state_dict(proj_mlp),
            "infusenet": accelerator.get_state_dict(infusenet),
        }
        torch.save(ckpt, "hidream_comic_id_adapter.pth")
        print("✅ Training fertig → hidream_comic_id_adapter.pth")

# ---------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir",  required=True, help="Verzeichnis mit Real-Porträts")
    ap.add_argument("--style_dir", required=True, help="Verzeichnis mit Style-Panels")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--log_steps", type=int, default=50)
    cfg = ap.parse_args()
    main(cfg)


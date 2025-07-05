# imdb_face_downloader.py
"""
Download the IMDb-Face dataset into a RAW_DIR/{dataset_name}/{id}/ structure,
mirroring the Modal workflow that is already used for Glint360K.

Prerequisites
-------------
* `aiohttp` (HTTP client), `tqdm` (progress bars) and `pandas`
  are available in the Modal image you already use for Glint360K.
* The utils module must expose the same helpers you used before:
  `copy_concurrent`, `ensure_dir`, `start_monitoring_disk_space`.
* `config.py` must define `RAW_DIR`, `app` and `raw_volume`
  exactly like in the existing project.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import logging
import random
import uuid
from pathlib import Path
from typing import List, Tuple
import gdown

import aiohttp
import modal
from tqdm.asyncio import tqdm_asyncio

from .config import RAW_DIR, app, raw_volume  # unchanged
from .utils import copy_concurrent, ensure_dir, start_monitoring_disk_space

# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------
DATASET_NAME = "imdb-face"
CSV_URL = (
    "https://drive.google.com/uc?export=download&id=134kOnRcJgHZ2eREu8QRi99qj996Ap_ML"
)  # 90 MB metadata file﻿:contentReference[oaicite:0]{index=0}
NUM_SHARDS = 100 # tune for cluster size / #workers
MAX_PER_ID = None               # safety-cap (set to None to keep everything)

LOCAL_DIR = Path("/tmp")        # tmpfs provided by Modal
HEADERS = {
    "User-Agent": "Mozilla/5.0 (IMDB-Face-Downloader)"  # nur ASCII
}

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _csv_path() -> Path:
    path = LOCAL_DIR / "imdb_face.csv"
    if not path.exists():
        gdown.download(CSV_URL, str(path), quiet=False, fuzzy=True)
    return path


def _shard_of(pid: str) -> int:
    """
    Konvertiert eine String-ID (z. B. 'nm0385722') in eine
    deterministische Ganzzahl und bildet sie auf [0, NUM_SHARDS).
    """
    # 128-Bit-Hash in int und Modulo
    return int(hashlib.md5(pid.encode("utf-8")).hexdigest(), 16) % NUM_SHARDS


def _iter_rows_for_shard(shard_idx: int) -> List[Tuple[str, str]]:
    """
    Liefert alle (person_id, image_url)-Paare, die zu `shard_idx` gehören.
    """
    rows: List[Tuple[str, str]] = []
    with _csv_path().open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["index"].strip()       # z. B. 'nm0385722'
            if _shard_of(pid) != shard_idx:
                continue
            url = row["url"].strip()
            if url:                          # manche Zeilen sind leer
                rows.append((pid, url))

    random.shuffle(rows)                     # gleichmäßiger Workload
    return rows

async def _download_one(
    session: aiohttp.ClientSession, dest: Path, url: str
) -> None:
    try:
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status == 404:
                return
            elif r.status != 200:
                logging.warning("✗ %s → HTTP %s", url, r.status)
                return

            img_bytes = await r.read()
            with dest.open("wb") as f:
                f.write(img_bytes)
    except Exception as ex:
        logging.warning("✗ %s → %s", url, ex)


# ----------------------------------------------------------------------
#  step 1 – download images for a single shard
# ----------------------------------------------------------------------
@app.function(
    volumes={RAW_DIR: raw_volume},
    timeout=60 * 60 * 12,           # 6 h per worker
    ephemeral_disk=512 * 1024,      # 512 GB tmpfs
    max_containers=10,
)
@modal.concurrent(max_inputs=4)
async def download_shard(shard: int) -> None:
    """
    Download all images whose identity_id % NUM_SHARDS == `shard`.

    The images are first written to a local tmpfs to keep egress fast,
    then copied into the RAW_DIR (Modal’s S3-backed volume) once the
    shard is finished.
    """
    print(f"### start shard {shard:04d}")
    start_monitoring_disk_space()

    # local scratch dir for this shard
    tmp_dir = LOCAL_DIR / f"{DATASET_NAME}_{shard:04d}"
    ensure_dir(tmp_dir)

    rows = _iter_rows_for_shard(shard)
    print(f"### {len(rows):,} images scheduled in shard {shard:04d}")

    # --- async HTTP download ------------------------------------------------
    connector = aiohttp.TCPConnector(limit=64)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(64)     # limit concurrent downloads

        async def wrapper(person_id: str, url: str):
            async with sem:
                if MAX_PER_ID and len(list((tmp_dir / person_id).glob("*.jpg"))) >= MAX_PER_ID:
                    return
                cls_dir = ensure_dir(tmp_dir / person_id)
                out_file = cls_dir / f"{uuid.uuid4()}.jpg"
                await _download_one(session, out_file, url)

        await tqdm_asyncio.gather(
            *[wrapper(pid, url) for pid, url in rows], desc=f"shard {shard:02d}"
        )

    # --- copy to persistent RAW_DIR ----------------------------------------
    print(f"### copying shard {shard:04d} to S3-backed volume")
    copy_concurrent(tmp_dir, Path(RAW_DIR) / DATASET_NAME)     # RAW_DIR is mounted at /mnt/…

    print(f"### done shard {shard:04d}")


# ----------------------------------------------------------------------
#  driver – launch the whole download with one Modal call
# ----------------------------------------------------------------------
@app.function(
    timeout=60 * 60 * 24,           # 24 h for coordinator
    max_containers=1,
)
def download_imdb_face() -> str:
    """
    Kick off `NUM_SHARDS` workers that each call `download_shard`.
    Returns the dataset name so you can pipe the result in larger workflows.
    """
    start_monitoring_disk_space()
    list(download_shard.map(range(NUM_SHARDS)))
    return DATASET_NAME

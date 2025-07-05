
from pathlib import Path

import modal
from config import RAW_DIR, app, raw_volume
from utils import copy_concurrent, ensure_dir, start_monitoring_disk_space

LOCAL_DIR = Path("/tmp")

# ----------------------------------------------------------------------
#  Schritt 1 – Shards der HF-Dataset laden
# ----------------------------------------------------------------------
@app.function(
    volumes={RAW_DIR: raw_volume},
    timeout=60 * 60 * 12,        # 6h pro Worker
    ephemeral_disk=512 * 1024,  # 512 GB tmpfs
    max_containers=8
)
@modal.concurrent(max_inputs=4)
async def download_shard(index: str) -> None:
    """
    Lädt einen Teil der HF-Shards (per allow_patterns) in ein lokales tmp-Verzeichnis.
    pattern z. B. 'glint360k-0[0-2]*.tar.gz' lädt die ersten ~30 Shards.
    """
    from pathlib import Path
    print (f"### start {index}")

    start_monitoring_disk_space()
    tmp_dir = LOCAL_DIR / index
    ensure_dir(tmp_dir)

    # Lädt einen TAR-Shard, behält die Original­auflösung bei und speichert
    # höchstens MAX_PER_ID Bilder pro Klassen-ID.


    import webdataset as wds

    start_monitoring_disk_space()

    url = f"https://huggingface.co/datasets/gaunernst/glint360k-wds-gz/resolve/main/glint360k-{index}.tar.gz"
    ds = wds.WebDataset(url,
        cache_dir=LOCAL_DIR,
        shardshuffle=False
    ).decode("pil").to_tuple("jpg", "cls")
    saved = 0


    for img, key in ds:
        cls = key
        cls_dir = ensure_dir(tmp_dir / f"{cls:06d}")
        import uuid
        uuid = uuid.uuid4()
        img_path = cls_dir / f"{uuid}.jpg"
        img.convert("RGB").save(img_path, format="JPEG", quality=95)
        saved += 1
        print(f"### ## saved {saved} for {index}   ")

    print(f"### ## start copying to s3 {index}")

    copy_concurrent(tmp_dir, Path("/mnt"))

    print(f"### ## end copying to s3 {index}")
    print (f"### end {index}")


@app.function(
    timeout=60 * 60 * 12,        # 6h pro Worker
    max_containers=1
)
def download_vgg2() -> str:
    dataset_name = "vgg2"
    start_monitoring_disk_space()
    list(download_shard.starmap(((str(i).zfill(4), dir)) for i in range(1385)))
    return dataset_name

from importlib.metadata import method_cache
# modal_laion_face_full.py
import shutil
import subprocess
from pathlib import Path
import modal

from .generate_style_pool import DATA_DIR, data_vol, APP_NAME
from sqlite3.dbapi2 import apilevel

app = modal.App(APP_NAME)

@app.function(
    image=modal.Image.debian_slim()
    .pip_install(
        "torch",
        "img2dataset==1.45.0",
        "pyarrow",  # für Parquet
        "wget",
    )
    .apt_install("wget")
    .env({
        "TMPDIR": "/data/tmp",
        "NO_ALBUMENTATIONS_UPDATE": "1"
    }),
    volumes={str(DATA_DIR): data_vol},
    timeout = 3600
)
def download():
    """
    Führt Schritte 1–4 automatisch aus:
     1) Metadata-Parquet herunterladen
     2) Face ID Liste & converter laden
     3) Face-Parquet erstellen
     4) Bilder downloaden flach ins output-Verzeichnis
    """
    parquet_url_base="https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/"
    face_ids_url="https://huggingface.co/datasets/FacePerceiver/laion-face/resolve/main/laion_face_ids.pth"
    converter_url="https://raw.githubusercontent.com/FacePerceiver/LAION-Face/master/convert_parquet.py"
    meta_dir: str =str(DATA_DIR / "laion_face_meta")
    output_dir=str(DATA_DIR / "training/real")
    num_images=50,
    num_workers=16,
    shard_start=0,
    shard_end=1,

    meta_path = Path(meta_dir)
    #if meta_path.exists():
    #    shutil.rmtree(meta_path)
    meta_path.mkdir(parents=True, exist_ok=True)

    tmp_path = Path("/data/tmp")
    tmp_path.mkdir(exist_ok=True)

    # 1) Metadata-Parquet herunterladen (rekursiv)
    subprocess.run([
        "wget", "--show-progress", "-P", meta_dir, "-q", "-l1", "-r", "-nd","-c", "-nc", "--no-parent",
        f"{parquet_url_base}" + "part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet"
    ], check=True)
    print("✅ Schritte 1 abgeschlossen: Parquet-Metadaten geholt.")

    # 2) FacePerceiver ID Liste & converter
    subprocess.run(["wget", "--show-progress", "-c", "-nc", "-P", meta_dir, "-q", face_ids_url], check=True)
    subprocess.run(["wget", "--show-progress", "-c", "-nc", "-P", meta_dir, "-q", converter_url], check=True)
    print("✅ Schritte 2 abgeschlossen: Face ID & converter geholt.")

    # 3) Face-only Parquet bauen
    # subprocess.run([
    #     "python3", "convert_parquet.py",
    #     meta_dir + "/" + Path(face_ids_url).name,
    #     meta_dir,
    #     meta_dir
    # ], check=True, cwd=meta_dir)
    # print("✅ Schritt 3 abgeschlossen: Face-only Parquet erstellt.")

    # 4) Bilder herunterladen
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    print("CWD:", os.getcwd())
    print("TMPDIR:", os.environ.get("TMPDIR"))
    print("Contents of /data:", os.listdir("/data"))
    print("out:", out)


    from img2dataset import download
    download(
        url_list="/data/laion_face_meta/laion_face_part_00000.parquet",
        input_format="parquet",
        url_col="URL",
        output_folder="/data/training/real",
        output_format="files",
        processes_count=1,
        thread_count=1,
        image_size=None,
        resize_mode="no",        # or a number like 256
        number_sample_per_shard=1,  # limits images per shard
        enable_wandb=False,
    )

    # cmd = [
    #     "img2dataset",
    #     "/data/laion_face_meta/laion_face_part_00000.parquet",
    #     "--input_format", "parquet",
    #     "--url_col", "URL",
    #     "--output_folder", "/data/training/real",
    #     "--output_format", "files",
    #     "--processes_count", "1",
    #     "--thread_count", "1",
    #     "--image_count", "50",
    #     "--enable_wandb",
    # ]
    # if shard_end is not None:
    #     cmd += ["--shard_count", "1"]
    # if shard_start:
    #     cmd += ["--start_shard", "0"]

    print("🔽 Starte Bild-Download...")
    #subprocess.run(cmd, check=True, shell=True, cwd=str(meta_dir))
    print("✅ Schritt 4 abgeschlossen: Bilder sind im Zielordner.")

# ---
# lambda-test: false  # long-running
# ---
#
# Glint360K ≈ 17 Mio Bilder, 360 k Identitäten
#
# Quelle (A) Academic Torrents – 7 TAR-Splits
# Quelle (B) Hugging Face – 1 385 WebDataset-Shards
#
# Dieses Skript lädt wahlweise (B), transformiert die Bilder auf 256×256
# und kopiert höchstens 20 Bilder pro Identity in einen S3-Bucket.
#


from .config import app
from cida.imdb_face import download_imdb_face

# ----------------------------------------------------------------------
#  Orchestrator
# ----------------------------------------------------------------------
@app.function(
    timeout=60 * 60 * 12
)
def start():

    # ----
# Download Datasets to raw
    # ----
    #from vgg2 import download_vgg2
    #datasets.append(download_vgg2.remote())

    datasets = list(download_imdb_face.map())

    # ----
    # Select Best Images and Prepare Datasets to raw
    # ----
    #from augment import process_dataset

    #process_dataset.map(datasets)

    print("Fertig!")

import modal

# ----------------------------------------------------------------------
#  Parameter
# ----------------------------------------------------------------------
MAX_PER_ID = 5
IMAGE_SIZE = 448


# ----------------------------------------------------------------------
#  modal - Infra
# ----------------------------------------------------------------------

bucket_creds = modal.Secret.from_name(
    "aws-secret", environment_name="main"
)

raw_bucket_name = "cida-datasets-raw"
target_bucket_name = "cida-datasets-target"

raw_volume = modal.CloudBucketMount(
    raw_bucket_name,
    secret=bucket_creds,
)

target_volume = modal.CloudBucketMount(
    target_bucket_name,
    secret=bucket_creds,
)

image = (
    modal.Image.debian_slim()
    .apt_install("wget", "git", "curl", "file")
    .pip_install(
        "webdataset~=0.2.33",  # zum Stream-Lesen von TAR-Shards
        "Pillow~=10.4",
        "huggingface_hub~=0.23.0",
        "gdown"
    )
)

RAW_DIR = "/mnt/raw"
TARGET_DIR = "/mnt/target"

app = modal.App("cida-prepare-dataset", image=image)

# select_best_faces.py
"""
Utility to iterate over a face–image dataset organised as
<base_dir>/<person_id>/<images>.jpg
and select the five most consistent images per person.

Changes in this version (July 2025)
-----------------------------------
* Bounding‑box‑Ermittlung basiert jetzt auf **MediaPipe Selfie Multiclass (256 × 256)**.
  Wir kombinieren die Kategorien *face skin* (3) und *hair* (1), ermitteln das
  umschließende Rechteck und **vergrößern die Unterkante um 10 %**, um den
  Halsansatz sicher einzuschließen.
* Klassische Face‑Detection ist entfallen.
* Der Rest (Facial‑Attribute‑Konsistenz, Greedy‑Auswahl, Crop auf 448 × 448)
  bleibt unverändert.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# MediaPipe – Vision Tasks
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import image_segmenter as mp_image_segmenter

# Externe Attribute‑Extraktion (z. B. Haarfarbe, Bart, Brille …)
from facial_attributes import FacialAttributesExtractor  # type: ignore


# ---------------------------------------------------------------------------
# Segmentation / Bounding‑Box‑Utilities
# ---------------------------------------------------------------------------

# Das Multi‑Class‑Modell liefert folgende Kategorie‑IDs (Stand Juli 2025):
# 0 = Background, 1 = Hair, 2 = Body‑Skin, 3 = Face‑Skin, 4 = Clothes …
_FACE_HAIR_LABELS = {1, 3}

# Pfad zum TFLite‑Modell (ggf. anpassen):
SELFIE_MODEL_PATH = "selfie_multiclass_256x256.tflite"

_BaseOptions = vision.BaseOptions
_SegmOptions = mp_image_segmenter.ImageSegmenterOptions

_segmenter_options = _SegmOptions(
    base_options=_BaseOptions(model_asset_path=SELFIE_MODEL_PATH),
    output_type=_SegmOptions.OutputType.CATEGORY_MASK,
)
_SEGMENTER = mp_image_segmenter.ImageSegmenter.create_from_options(_segmenter_options)


def _bbox_from_segmentation(mask: np.ndarray, down_expand: float = 0.10) -> Tuple[int, int, int, int] | None:
    """Ermittle Bounding‑Box der *face*+*hair* Klassen und erweitere sie nach unten."""
    ys, xs = np.where(np.isin(mask, list(_FACE_HAIR_LABELS)))
    if xs.size == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    h_box = y2 - y1
    y2 = min(mask.shape[0] - 1, y2 + int(h_box * down_expand))
    return int(x1), int(y1), int(x2), int(y2)


def _segment_image(img_rgb: np.ndarray) -> np.ndarray:
    """Segmentiere Bild und liefere Category‑Mask als NumPy‑Array."""
    mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=img_rgb)
    result = _SEGMENTER.segment(mp_image)
    mask = result.category_mask.numpy_view()
    return mask


# ---------------------------------------------------------------------------
# Konsistenz‑Scoring (unverändert)
# ---------------------------------------------------------------------------

class ConsistencyScorer:
    WEIGHTS = {
        "hair_color": 1.0,
        "hair_length_cm": 0.5,
        "beard": 0.8,
        "moustache": 0.8,
        "glasses": 0.6,
    }

    def __call__(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        dist = 0.0
        for k, w in self.WEIGHTS.items():
            if k not in a or k not in b:
                continue
            if isinstance(a[k], (bool, str)):
                dist += 0.0 if a[k] == b[k] else w
            else:
                dist += w * min(1.0, abs(float(a[k]) - float(b[k])) / 10.0)
        return dist


# ---------------------------------------------------------------------------
# Daten‑Container
# ---------------------------------------------------------------------------

class ImageInfo:
    def __init__(self, path: Path, img: np.ndarray, bbox: Tuple[int, int, int, int], attrs: Dict[str, Any]):
        self.path = path
        self.img = img
        self.bbox = bbox
        self.attrs = attrs

    def crop(self, size: int = 448) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        crop = self.img[y1 : y2, x1 : x2]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Auswahl der besten Bilder (Greedy‑Heuristik)
# ---------------------------------------------------------------------------

def _select_best(img_infos: List[ImageInfo], k: int = 5) -> List[ImageInfo]:
    if len(img_infos) <= k:
        return img_infos
    scorer = ConsistencyScorer()
    n = len(img_infos)
    dists = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = scorer(img_infos[i].attrs, img_infos[j].attrs)
            dists[i, j] = dists[j, i] = d
    chosen = [int(dists.mean(axis=1).argmin())]
    while len(chosen) < k:
        rem = [i for i in range(n) if i not in chosen]
        idx = min(rem, key=lambda r: sum(dists[r, c] for c in chosen))
        chosen.append(idx)
    return [img_infos[i] for i in chosen]


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------
from config import app, RAW_DIR, TARGET_DIR

@app.function(
    timeout=60 * 60 * 12
)
def process_dataset(dataset_name: str) -> None:
    base_dir = Path(RAW_DIR) / dataset_name
    output_dir = Path(TARGET_DIR) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    attr_extractor = FacialAttributesExtractor()

    for person_dir in sorted(d for d in base_dir.iterdir() if d.is_dir()):
        person_id = person_dir.name
        infos: List[ImageInfo] = []

        for img_path in sorted(person_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = _segment_image(img_rgb)
            bbox = _bbox_from_segmentation(mask)
            if bbox is None or (bbox[3] - bbox[1]) < 100:
                continue
            x1, y1, x2, y2 = bbox
            attrs = attr_extractor.predict(img_rgb[y1:y2, x1:x2])
            infos.append(ImageInfo(img_path, img, bbox, attrs))

        if not infos:
            print(f"[WARN] {person_id}: keine geeigneten Bilder gefunden")
            continue

        best = _select_best(infos, 5)
        out_dir = output_dir / person_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, info in enumerate(best, 1):
            cv2.imwrite(str(out_dir / f"img_{i:02d}.jpg"), info.crop())
        print(f"[INFO] {person_id}: {len(best)} Bilder → {out_dir}")

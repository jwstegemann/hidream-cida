#!/usr/bin/env python
"""
Comic-Augmentation mit HED-Konturen
  – HED liefert geschlossene, saubere Umrisslinien
  – anschließend Morph Close + Konturenfilter
  – Resultat = flache Farbflächen + schwarze Outlines
"""

import argparse, random, pathlib, cv2, numpy as np

# einmalig laden
proto  = "models/hed/deploy.prototxt"
model  = "models/hed/hed_pretrained_bsds.caffemodel"
net    = cv2.dnn.readNetFromCaffe(proto, model)

# ───────────── Bild-Utilities ──────────────────────────────
def unsharp(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.addWeighted(img, 1.3, blur, -0.3, 0)


def mean_shift(img, sp=15, sr=25):
    return cv2.pyrMeanShiftFiltering(img, sp, sr)


def kmeans_quant(img_bgr, k=6):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    data = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _ret, labels, centers = cv2.kmeans(
        data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    quant = centers[labels.flatten()].reshape((h, w, 3)).astype(np.uint8)
    return cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)


# ───────────── HED-Kantenerkennung ─────────────────────────
def hed_edges(img_bgr, net, min_len=80, thick=3):
    """
    Liefert uint8-Maske (255 = Linie) ohne Versatz
    """
    h,  w  = img_bgr.shape[:2]
    pad_h  = (-h) % 16              # 0…15
    pad_w  = (-w) % 16

    # 1️⃣ symmetrisch auffüllen (Reflektieren vermeidet harte Kanten)
    img_pad = cv2.copyMakeBorder(
        img_bgr, 0, pad_h, 0, pad_w,
        cv2.BORDER_REFLECT_101
    )
    H, W = img_pad.shape[:2]

    # 2️⃣ HED-Forward
    blob = cv2.dnn.blobFromImage(
        img_pad,
        scalefactor=1.0,
        size=(W, H),                # exakt aufgepolsterte Größe
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False
    )
    net.setInput(blob)
    edged = net.forward()[0, 0]     # (H, W), float32 0…1

    # 3️⃣ Padding wieder abschneiden → Originalmaß
    edged = edged[:h, :w]
    edged = (edged * 255).astype(np.uint8)

    # 4️⃣ binarisieren + Konturen filtern
    _, bw  = cv2.threshold(edged, 50, 255, cv2.THRESH_BINARY_INV)
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw     = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    mask   = np.zeros_like(bw)
    cnts, _= cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        if cv2.arcLength(c, True) >= min_len:
            cv2.drawContours(mask, [c], -1, 255, thick, lineType=cv2.LINE_AA)
    return mask


def hsv_jitter(img_bgr, h_deg=5, s=0.1, v=0.1):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s_, v_ = cv2.split(hsv)
    h = (h + random.uniform(-h_deg, h_deg)) % 180
    s_ *= 1 + random.uniform(-s, s)
    v_ *= 1 + random.uniform(-v, v)
    hsv_aug = cv2.merge((h, s_.clip(0, 255), v_.clip(0, 255))).astype(np.uint8)
    return cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)


# ───────────── Haupt-Pipeline ───────────────────────────────
def comicify(img_bgr, k=6, hed_dir="models/hed", min_len=80, jitter=True):
    sharp = unsharp(img_bgr)
    flat  = mean_shift(sharp)
    quant = kmeans_quant(flat, k)

    edges  = hed_edges(quant, net, min_len=80, thick=3)
    comic  = quant.copy()
    comic[edges == 255] = (0, 0, 0)

    return hsv_jitter(comic) if jitter else comic


# ───────────── CLI ─────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Comic-Augment mit HED-Konturen")
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--k", type=int, default=6, help="Farbcluster (4-8)")
    ap.add_argument("--hed", default="models/hed", help="Pfad zu HED-Modellordner")
    ap.add_argument("--minlen", type=int, default=80,
                    help="Minimale Konturlänge für Filterung")
    ap.add_argument("--no-jitter", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.input)

    out = comicify(
        img, k=args.k, hed_dir=args.hed,
        min_len=args.minlen, jitter=not args.no_jitter
    )
    cv2.imwrite(args.output, out)
    print("✅  Ergebnis gespeichert:", args.output)

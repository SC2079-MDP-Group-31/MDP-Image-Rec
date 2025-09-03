from PIL import Image
import io, uuid, os
import cv2
import numpy as np
from dotenv import load_dotenv
import yaml
import torch
import json
from PIL import Image, ImageOps
from ultralytics import YOLO

# load variables from config file into the environment
with open("app/config.yaml", "r") as f:   # adjust path if config.yaml is elsewhere
    cfg = yaml.safe_load(f)

# --- config / one-time load ---
MODEL_PATH = cfg["model"]   # Model name
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = int(cfg.get("imgsz", 640))
CONF = float(cfg.get("conf", 0.5))
IOU = float(cfg.get("iou", 0.5))
JPEG_QUALITY = int(cfg.get("jpeg_quality", 95))
LINE_WIDTH = int(cfg.get("line_width", 2))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model = YOLO(MODEL_PATH).to(DEVICE)

def _bytes_to_rgb_numpy(image_bytes: bytes) -> np.ndarray:
    """Decode bytes -> contiguous uint8 RGB HxWx3, with EXIF orientation fixed."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")  # handle orientation + force RGB
    arr = np.asarray(img, dtype=np.uint8)
    return np.ascontiguousarray(arr)

def model_predict_download( image_bytes: bytes):
    # 1) Decode bytes to numpy
    img_rgb = _bytes_to_rgb_numpy(image_bytes)            # (H,W,3) RGB

    # 2) Inference (Ultralytics accepts numpy arrays directly)
    res = model.predict(
        source=img_rgb,
        conf=CONF,
        iou=IOU,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )[0]

    # 3) Render annotated image (BGR numpy)
    annotated_bgr = res.plot(line_width=LINE_WIDTH)

    # 4) Convert to RGB for saving via PIL (or keep BGR and use cv2.imencode)
    annotated_bgr = res.plot(line_width=LINE_WIDTH)
    ok, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")

    buffer = io.BytesIO(buf.tobytes())
    buffer.seek(0)
    filename = f"prediction_{uuid.uuid4().hex[:8]}.jpg"
    return buffer, filename
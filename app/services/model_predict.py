from PIL import Image
import io
import cv2
import numpy as np
import os
from dotenv import load_dotenv
import yaml
import torch
import uuid
import json
from ultralytics import YOLO

# load variables from config file into the environment
with open("app/config.yaml", "r") as f:   # adjust path if config.yaml is elsewhere
    cfg = yaml.safe_load(f)

# --- config / one-time load ---
MODEL_PATH = cfg["model"]   # Model name
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = cfg.get("imgsz", 640)

model = YOLO(MODEL_PATH).to(DEVICE)

def _bytes_to_rgb_numpy(image_bytes: bytes) -> np.ndarray:
    """Decode bytes -> RGB numpy array (H,W,3). Handles PNG/JPG."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)

def model_predict_download(
    image_bytes: bytes,
    *,
    conf: float = cfg.get("conf", 0.5),
    iou: float = cfg.get("iou", 0.5),
    classes: list[int] | None = None,
    line_width: int = 2,
    jpeg_quality: int = 90,
):
    # 1) Decode bytes to numpy
    img_rgb = _bytes_to_rgb_numpy(image_bytes)            # (H,W,3) RGB

    # 2) Inference (Ultralytics accepts numpy arrays directly)
    res = model.predict(
        source=img_rgb,
        conf=conf,
        iou=iou,
        imgsz=IMG_SIZE,
        classes=classes,
        device=DEVICE,
        verbose=False
    )[0]

    # 3) Render annotated image (BGR numpy)
    annotated_bgr = res.plot(line_width=line_width)

    # 4) Convert to RGB for saving via PIL (or keep BGR and use cv2.imencode)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # 5) Encode to JPEG in-memory
    buffer = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)

    filename = f"prediction_{uuid.uuid4().hex[:8]}.jpg"
    return buffer, filename
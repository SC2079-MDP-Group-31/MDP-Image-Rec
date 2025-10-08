import io, uuid, os
import debugpy
import cv2
import numpy as np
from dotenv import load_dotenv
import yaml
import torch
import json
from PIL import Image, ImageOps
from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

# load variables from config file into the environment
with open("app/config.yaml", "r") as f:   # adjust path if config.yaml is elsewhere
    cfg = yaml.safe_load(f)

# --- config / one-time load ---
MODEL_PATH = cfg["model"]   # Model name
PREDICTIONS_DIR = cfg.get("predictions_dir", "outputs/predictions")
STITCHED_DIR = cfg.get("stitched_dir", "outputs/stitched")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = int(cfg.get("imgsz", 640))
CONF = float(cfg.get("conf", 0.5))
IOU = float(cfg.get("iou", 0.5))
JPEG_QUALITY = int(cfg.get("jpeg_quality", 95))
LINE_WIDTH = int(cfg.get("line_width", 2))
FONT_SIZE = int(cfg.get("font_size", 5))
MIN_AREA = int(cfg.get("min_area", 3200))  # minimum area (in pixels) for valid detection box
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def _bytes_to_rgb_numpy(image_bytes: bytes) -> np.ndarray:
    """Decode bytes -> contiguous uint8 RGB HxWx3, with EXIF orientation fixed."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")  # handle orientation + force RGB

    # Flip image 180 degrees for camera orientation
    flipped_img = img.rotate(180) 
    arr = np.asarray(flipped_img, dtype=np.uint8)

    # Dont flip image 180 degrees for camera orientation
    # arr = np.asarray(img, dtype=np.uint8)

    return np.ascontiguousarray(arr)

# Function to handle image prediction and return annotated image for download
def model_predict_download(image_bytes: bytes, returnJSON: bool = False, model_path: str = MODEL_PATH):

    print('Using model located in', model_path)
    # Load in model
    model = YOLO(model_path).to(DEVICE)

    # 1) Decode bytes to numpy
    img_rgb = _bytes_to_rgb_numpy(image_bytes)            # (H,W,3) RGB
    
    # Convert to grayscale with 3 channels
    img_rgb_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_rgb_gray2 = cv2.cvtColor(img_rgb_gray, cv2.COLOR_GRAY2RGB)

    # 2) Inference (Ultralytics accepts numpy arrays directly)
    res = model.predict(
        source=img_rgb_gray2,
        conf=CONF,
        iou=IOU,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False,
    )[0]

    detectionBoxs = res.boxes
    
    # Filter out detections with less than MIN_AREA
    if len(detectionBoxs) > 0:
        print(f"Filtering out detection boxes with area less than {MIN_AREA} pixels...")
        areas = detectionBoxs.xywh[:, 2] * detectionBoxs.xywh[:, 3]

        # Boolean mask for detections above threshold
        mask = areas >= MIN_AREA

        # keep only detectionBoxes with area >= MIN_AREA
        detectionBoxs = detectionBoxs[mask]

        # Reassign filtered boxes back to result
        res.boxes = detectionBoxs
    
    # Filter out all detections less the one with the largest bounding box
    if len(detectionBoxs) >= 2:
        print("More than 1 object detected in the image....selecting closer object")
        # Compute area for each detection
        areas = detectionBoxs.xywh[:, 2] * detectionBoxs.xywh[:, 3]
        # Find index of largest area
        largest_idx = int(torch.argmax(areas))

        # Keep only that detection
        largest_box_data = res.boxes.data[largest_idx].unsqueeze(0)  # tensor shape (1,6)
        res.boxes = Boxes(largest_box_data, res.boxes.orig_shape)    # reconstruct single-box object


    # 3) Render annotated image (BGR numpy)
    res.orig_img = img_rgb.copy() # Restore original image for annotation
    annotated_bgr = res.plot(line_width=LINE_WIDTH, font_size=FONT_SIZE)

    # 4) Convert to RGB for saving via PIL (or keep BGR and use cv2.imencode)
    annotated_bgr = res.plot(line_width=LINE_WIDTH, font_size=FONT_SIZE)
    ok, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")

    buffer = io.BytesIO(buf.tobytes())
    buffer.seek(0)
    filename = f"prediction_{uuid.uuid4().hex[:8]}.jpg"

    # 3) Parse results
    # First path is for default/best model, else statement for backup model
    if MODEL_PATH == "models/task1/task1.pt":
        predictions = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id] if model.names and cls_id in model.names else str(cls_id)
            conf = float(box.conf[0])
            predictions.append({
                "class_id": cls_id,
                "image_id": cls_id + 11,
                "confidence": conf,
            })
    else:
        predictions = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id] if model.names and cls_id in model.names else str(cls_id)
            conf = float(box.conf[0])
            predictions.append({
                "class_id": cls_id,
                "image_id": cls_id + 11,
                "confidence": conf,
            })

    predictions = {"predictions": predictions}

    return buffer, filename, predictions

# Function to handle image prediction and return results as JSON
def model_predict(image_bytes: bytes):
    # 1) Decode bytes to numpy
    img_rgb = _bytes_to_rgb_numpy(image_bytes)            # (H,W,3) RGB

    # 2) Inference (Ultralytics accepts numpy arrays directly)
    res = model.predict(
        source=img_rgb,
        conf=CONF,
        iou=IOU,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False,
    )[0]

    # 3) Parse results
    predictions = []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id] if model.names and cls_id in model.names else str(cls_id)
        conf = float(box.conf[0])
        predictions.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
        })

    return {"predictions": predictions}

def stitch_img():
    # Resolve paths
    image_files = sorted(Path(PREDICTIONS_DIR).glob("*.jpg"))
    filename = image_files[0].stem + "_stitched.jpg"
    save_path = os.path.join(STITCHED_DIR, filename)
    
    if not image_files:
        raise FileNotFoundError("No images found in outputs directory")

    # open images
    images = [Image.open(img) for img in image_files]

    # horizontal stitch
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    stitched = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in images:
        stitched.paste(img, (x_offset, 0))
        x_offset += img.width
    
    # Save to file
    stitched.save(save_path)
    print(f"✅ Stitched image saved to: {save_path}")

    return stitched
import debugpy
import os
import yaml
from pathlib import Path
from PIL import Image
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from services.model_predict import *
from services.pathing_algo import *
router = APIRouter()

# ------------------------ LOAD CONFIGS --------------------------------
# load variables from config file into the environment
with open("app/config.yaml", "r") as f:   # adjust path if config.yaml is elsewhere
    cfg = yaml.safe_load(f)

PREDICTIONS_DIR = cfg.get("predictions_dir", "outputs/predictions")

# ----------------------- TEST APIS --------------------------------------------------
# Simple test endpoint to verify server connectivity
@router.get("/test")
async def test():
    try:
        response = "Server Connected!"
        return response
    except Exception as e:
        print("❌ Internal server error:", e)
        return {"error": str(e)}

# This endpoint handles image uploads and returns predictions as JSON
@router.post("/predict-json")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        results = model_predict(contents)
        return results

    except Exception as e:
        print("❌ Internal server error:", e)
        return {"error": str(e)}

# This endpoint handles image uploads and returns the image with predictions for download
@router.post("/predict-download-test")
async def predict_download_image(file: UploadFile = File(...)):
    contents = await file.read()
    buffer, filename, predictions = model_predict_download(contents)

    return StreamingResponse(
        buffer,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

# ------------------- PRODUCTION APIS --------------------------------------------------
# This endpoint handles image uploads and saves the predicted image on the server locally while returning the predicted data as JSON
@router.post("/image")
async def predict_save_image(file: UploadFile = File(...)):
    print("Request Received To Predict Image.")
    contents = await file.read()
    buffer, filename, predictions = model_predict_download(contents)

    # ---- Save file to server directory ----
    save_path = os.path.join(PREDICTIONS_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(buffer.getbuffer())  # buffer is BytesIO, so use getbuffer()
    

    return predictions

# This endpoint handles image stitching (if needed) and displays the stitched image
@router.get("/stitch")
async def stitch_images():
    stitched_img = stitch_img()
    stitched_img.show()  # This will open the stitched image using the default image viewer
    return {"message": "Stitched image displayed."}

# This endpoint obtains the obstacle data and returns the robot pathing
@router.post("/path")
async def get_prediction_path(raw_data: str):
    commands = run_minimal(raw_data)
    return {"predictions_path": commands}
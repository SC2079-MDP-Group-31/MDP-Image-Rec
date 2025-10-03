import debugpy
import os
import yaml
from pathlib import Path
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
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
@router.post("/path-test")
async def get_prediction_path_with_coordinates_test(raw_data: str):
    from services.pathing_algo import run_minimal_with_coordinates
    commands_with_coords = run_minimal_with_coordinates(raw_data)
    
    # Format the response to include both commands and coordinates
    formatted_response = []
    for cmd_str, position in commands_with_coords:
        if position is not None:
            formatted_response.append({
                "commands": cmd_str,
                "estimated_position": {
                    "x": position.x//10,
                    "y": position.y//10,
                    "d": {
                        "TOP": 0,
                        "RIGHT": 1,
                        "BOTTOM": 2,
                        "LEFT": 3
                    }.get(position.direction.name if hasattr(position.direction, 'name') else str(position.direction), position.direction)
                }
            })
        else:
            # Handle case where position tracking is not available
            formatted_response.append({
                "command": cmd_str,
                "estimated_position": None
            })
    
    commands = []
    path = [{
      "x": 1,
      "y": 1,
      "d": 0
    },]
    for item in formatted_response:
        commands.append(item['commands'])   
        path.append(item['estimated_position'])

    print(f"Formatted response: {formatted_response}")

    return {
        "commands": commands,
        "path": path,
        "total_commands": len(formatted_response)
    }

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
# Simple test endpoint to verify server connectivity
@router.get("/status")
async def test():
    try:
        response = "Server Connected!"
        return response
    except Exception as e:
        print("❌ Internal server error:", e)
        return {"error": str(e)}

# This endpoint handles image uploads and saves the predicted image on the server locally while returning the predicted data as JSON
@router.post("/image")
async def predict_save_image(file: UploadFile = File(...)):
    print("Request Received To Predict Image.")
    contents = await file.read()
    buffer, filename, predictions = model_predict_download(contents)
    print("Image prediction completed")
    # ---- Save file to server directory ----
    save_path = os.path.join(PREDICTIONS_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(buffer.getbuffer())  # buffer is BytesIO, so use getbuffer()
    print("Predicted image saved to server as ", filename)
    
    return predictions

# This endpoint handles image stitching (if needed) and displays the stitched image
@router.get("/stitch")
async def stitch_images():
    stitched_img = stitch_img()
    stitched_img.show()  # This will open the stitched image using the default image viewer
    return {"message": "Stitched image displayed."}

# This endpoint obtains the obstacle data and returns the robot pathing
# @router.post("/path")
# async def get_prediction_path(raw_data: str):
#     commands = run_minimal(raw_data)
#     return {"predictions_path": commands}

# This endpoint obtains the obstacle data and returns the robot pathing with coordinates
@router.post("/path")
async def get_prediction_path_with_coordinates(request: Request):
    # Require text/plain
    ct = (request.headers.get("content-type") or "").lower()
    if "text/plain" not in ct:
        raise HTTPException(415, detail="Send raw text with Content-Type: text/plain")
    
    raw_data = (await request.body()).decode("utf-8").strip()

    print(f"Raw data received at /path endpoint: {raw_data}")
    from services.pathing_algo import run_minimal_with_coordinates
    commands_with_coords = run_minimal_with_coordinates(raw_data)
    
    # Format the response to include both commands and coordinates
    formatted_response = []
    for cmd_str, position in commands_with_coords:
        if position is not None:
            formatted_response.append({
                "commands": cmd_str,
                "estimated_position": {
                    "x": position.x//10,
                    "y": position.y//10,
                    "d": {
                        "TOP": 0,
                        "RIGHT": 1,
                        "BOTTOM": 2,
                        "LEFT": 3
                    }.get(position.direction.name if hasattr(position.direction, 'name') else str(position.direction), position.direction)
                }
            })
        else:
            # Handle case where position tracking is not available
            formatted_response.append({
                "command": cmd_str,
                "estimated_position": None
            })
    
    commands = []
    path = [{
      "x": 1,
      "y": 1,
      "d": 0
    },]
    for item in formatted_response:
        commands.append(item['commands'])   
        path.append(item['estimated_position'])

    print(f"Formatted response: {formatted_response}")

    return {
        "commands": commands,
        "path": path,
        "total_commands": len(formatted_response)
    }
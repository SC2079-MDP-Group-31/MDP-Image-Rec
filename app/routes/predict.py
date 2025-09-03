from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from app.services.model_predict import model_predict_download
router = APIRouter()

@router.get("/test")
async def test():
    try:
        response = "Server Connected!"
        return response
    except Exception as e:
        print("❌ Internal server error:", e)
        return {"error": str(e)}

# This endpoint handles image uploads and returns predictions as JSON
'''
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        results = model_predict(contents)
        return results

    except Exception as e:
        print("❌ Internal server error:", e)
        return {"error": str(e)}
'''

# This endpoint handles image uploads and returns the image with predictions for download
@router.post("/predict-download")
async def predict_download_image(file: UploadFile = File(...)):
    contents = await file.read()
    buffer, filename = model_predict_download(contents)

    return StreamingResponse(
        buffer,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )
from fastapi import FastAPI
from app.routes import predict

app = FastAPI(title="Ingredient Detector")

# Include route modules
app.include_router(predict.router, prefix="/api", tags=["Predict"])
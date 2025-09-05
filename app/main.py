from fastapi import FastAPI
import uvicorn
from routes import predict

app = FastAPI(title="MDP Obstacle Recognition")

# Include route modules
app.include_router(predict.router, prefix="/image", tags=["Predict"])

if __name__ == "__main__":
    # exposes on all interfaces; change host/port as needed
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)